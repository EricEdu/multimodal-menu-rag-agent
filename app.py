import csv
import io
import json
import os
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(page_title="Assistente Inteligente de Cardápio", page_icon="🍔", layout="centered")

st.title("🍔 Assistente Inteligente de Cardápio (Multimodal RAG)")
st.caption("Ingestão de fotos de cardápios via Visão + Structured Outputs, busca vetorial e atendimento seguro.")

class ItemCardapio(BaseModel):
    id: str = Field(description="Código sequencial curto como ITEM-01")
    nome: str = Field(description="Nome do prato ou bebida")
    categoria: str = Field(description="Categoria como Porções, Pratos Principais, Bebidas, etc.")
    preco: float = Field(description="Preço numérico em reais")
    descricao: str = Field(description="Ingredientes ou descrição do prato")
    alergenicos: list[str] = Field(description="Lista de possíveis alergênicos como glúten, lactose, peixe, soja, etc.")
    restricoes: list[str] = Field(description="Lista de dietas compatíveis como vegetariano, vegano, sem lactose, sem glúten")
    tempo_preparo_min: int = Field(default=20, description="Tempo estimado de preparo em minutos")

class CardapioExtraido(BaseModel):
    itens: list[ItemCardapio]

api_key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input("Insira sua GEMINI_API_KEY", type="password")

st.sidebar.divider()
st.sidebar.subheader("📂 Base de Conhecimento")
arquivo_upload = st.sidebar.file_uploader(
    "Envie uma FOTO de cardápio ou arquivo (.png, .jpg, .json, .csv, .txt)",
    type=["png", "jpg", "jpeg", "webp", "json", "csv", "txt"],
    help="Envie uma imagem para extração automática via IA ou um arquivo estruturado."
)

def formatar_item(item: dict) -> str:
    alerg = ", ".join(item.get("alergenicos", [])) if isinstance(item.get("alergenicos"), list) else item.get("alergenicos", "Nenhum")
    rest = ", ".join(item.get("restricoes", [])) if isinstance(item.get("restricoes"), list) else item.get("restricoes", "Nenhuma")
    preco = float(item.get("preco", 0))
    return (
        f"ID: {item.get('id', 'N/A')} | Nome: {item.get('nome', 'Item')} ({item.get('categoria', 'Geral')}) | "
        f"Preço: R$ {preco:.2f} | Tempo: {item.get('tempo_preparo_min', 20)} min | "
        f"Descrição: {item.get('descricao', '')} | Alergênicos: {alerg or 'Nenhum'} | Dietas: {rest or 'Nenhuma'}"
    )

@st.cache_data(show_spinner="🔍 Lendo imagem do cardápio e estruturando JSON com Gemini Vision...")
def extrair_cardapio_da_imagem(_client, bytes_imagem: bytes, mime_type: str) -> list[dict]:
    prompt_extracao = (
        "Extraia todos os itens, categorias, preços e descrições visíveis nesta imagem de cardápio. "
        "Com base nos ingredientes visíveis ou implícitos no nome do prato (ex: queijo -> lactose, empanado/pão -> glúten, peixe/tilápia -> peixe), "
        "preencha de forma conservadora os alergênicos e restrições alimentares."
    )
    resposta = _client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            types.Part.from_bytes(data=bytes_imagem, mime_type=mime_type),
            prompt_extracao,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CardapioExtraido,
            temperature=0.1,
        ),
    )
    dados = json.loads(resposta.text)
    return dados.get("itens", [])

def carregar_itens(client, arquivo) -> tuple[list[str], list[dict] | None]:
    if arquivo is None:
        with open("cardapio.json", "r", encoding="utf-8") as f:
            dados = json.load(f)
            return [formatar_item(item) for item in dados], dados

    nome = arquivo.name.lower()
    if nome.endswith((".png", ".jpg", ".jpeg", ".webp")):
        dados = extrair_cardapio_da_imagem(client, arquivo.getvalue(), arquivo.type)
        return [formatar_item(item) for item in dados], dados

    conteudo = arquivo.getvalue().decode("utf-8")
    if nome.endswith(".json"):
        dados = json.loads(conteudo)
        return [formatar_item(item) for item in dados], dados
    elif nome.endswith(".csv"):
        leitor = csv.DictReader(io.StringIO(conteudo))
        dados = list(leitor)
        return [formatar_item(linha) for linha in dados], dados
    else:
        linhas = [linha.strip() for linha in conteudo.splitlines() if linha.strip()]
        return linhas, None

@st.cache_resource
def gerar_indice_vetorial(_client, itens_texto: tuple[str, ...]):
    resposta = _client.models.embed_content(
        model="gemini-embedding-001",
        contents=list(itens_texto),
    )
    vetores = np.array([emb.values for emb in resposta.embeddings], dtype=np.float32)
    normas = np.linalg.norm(vetores, axis=1, keepdims=True)
    return vetores / np.maximum(normas, 1e-10)

def buscar_contexto_relevante(client, pergunta: str, itens_texto: list[str], matriz_vetores: np.ndarray, top_k: int = 6) -> str:
    resp_query = client.models.embed_content(
        model="gemini-embedding-001",
        contents=pergunta,
    )
    vetor_q = np.array(resp_query.embeddings[0].values, dtype=np.float32)
    vetor_q = vetor_q / max(np.linalg.norm(vetor_q), 1e-10)
    similaridades = matriz_vetores @ vetor_q
    k = min(top_k, len(itens_texto))
    indices_top = np.argsort(similaridades)[::-1][:k]
    return "\n".join([itens_texto[i] for i in indices_top])

if not api_key:
    st.info("👈 Adicione sua chave da API do Google Gemini na barra lateral para testar o agente.")
    st.stop()

client = genai.Client(api_key=api_key)
itens_formatados, dados_json = carregar_itens(client, arquivo_upload)

st.sidebar.success(f"✅ {len(itens_formatados)} itens indexados na memória vetorial.")
if dados_json:
    st.sidebar.download_button(
        label="⬇️ Baixar JSON Estruturado",
        data=json.dumps(dados_json, ensure_ascii=False, indent=2),
        file_name="cardapio_extraido.json",
        mime="application/json"
    )

matriz_embeddings = gerar_indice_vetorial(client, tuple(itens_formatados))

if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        {"role": "assistant", "content": "Olá! Sou o assistente do restaurante. Envie a foto de um cardápio na barra lateral ou pergunte sobre os itens disponíveis!"}
    ]

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: Quais porções sem lactose vocês têm na foto e quanto custam 2 bebidas?"):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    contexto_recuperado = buscar_contexto_relevante(client, prompt, itens_formatados, matriz_embeddings, top_k=6)

    system_instruction = f"""
    Você é um assistente especialista de atendimento de delivery.
    Responda APENAS com base nos itens do cardápio fornecidos no contexto abaixo.
    
    Regras estritas de segurança e precisão:
    1. Nunca invente pratos, preços ou ingredientes que não estejam no contexto.
    2. Se o cliente mencionar alergias ou restrições (ex: sem glúten, vegano, sem lactose), verifique rigorosamente os campos 'Alergênicos' e 'Dietas' antes de recomendar.
    3. Sempre que o cliente pedir uma combinação de itens, calcule e informe o valor total exato em Reais (R$) e o tempo estimado máximo de preparo.
    4. Se nenhum item atender ao pedido, informe educadamente e sugira a alternativa mais próxima disponível.

    CONTEXTO DO CARDÁPIO RECUPERADO:
    {contexto_recuperado}
    """

    with st.chat_message("assistant"):
        resposta = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        st.markdown(resposta.text)
        with st.expander("🔍 Ver contexto RAG recuperado (Debug Técnico)"):
            st.text(contexto_recuperado)

    st.session_state.mensagens.append({"role": "assistant", "content": resposta.text})
