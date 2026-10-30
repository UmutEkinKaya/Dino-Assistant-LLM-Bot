"""Bütün şemalar: modelin üreteceği cevaplar ve API'nin sözleşmesi.

İki tür şema var ve ikisi de pydantic:
  1) LLM ÇIKTI şemaları (LayeredAnswer, FactCheck): description'lar prompt'un parçasıdır.
     Alan sırası önemli: gerekçe/kanıt alanları karar alanından ÖNCE gelir (model önce kanıt
     yazar, sonra karar verir).
  2) API şemaları (AskRequest, ...): dış dünyayla sözleşme. Kurala uymayan istek fonksiyona
     ulaşmadan 422 ile reddedilir.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ==========================================================================================
# 1) LLM çıktı şemaları
# ==========================================================================================
class Claim(BaseModel):
    text: str = Field(description="ONE short fact or conclusion in your own words, max 25 words")
    sources: list[str] = Field(description="Chunk ids exactly as shown in brackets, e.g. guide:s4.2 or tyrannosaurus:72, or a tool name such as "
                                           "find_species or compare_timeline")


class LayeredAnswer(BaseModel):
    """Lab'daki şemalı_v2'nin devamı: belge / çıkarım / genel bilgi ayrı katmanlarda."""
    corrections: list[str] = Field(description="If the question assumes something the sources "
                                               "contradict, state the correction here and ONLY here")
    document: list[Claim] = Field(description="Only facts from the tool results that help answer "
                                              "THIS question")
    inference: list[Claim] = Field(description="Conclusions from combining several facts")
    not_in_document: list[str] = Field(description="Parts of the question the sources do not cover")
    general_knowledge: list[str] = Field(description="Widely known facts about the uncovered parts, "
                                                     "NOT from the sources. Empty if not useful.")
    summary: str = Field(description="2-3 sentences in the language of the question, based ONLY on "
                                     "corrections, document and inference. If general knowledge was "
                                     "used, end with a separate sentence starting 'Belge dışı bilgi:' "
                                     "or 'Outside the document:'.")


class GeneralFact(BaseModel):
    """Belge dışı bilgi + doğrulama sorgusu. Model URL YAZMAZ (uydurulmuş link riski);
    sadece arama kelimelerini verir, linki kod kurar (Wikipedia araması: her zaman çalışır)."""
    text: str
    search_query: str = Field(description="2-5 English keywords to verify this fact on Wikipedia, "
                                          "e.g. 'Ceratosaurus length'")


class LayeredAnswerV2(BaseModel):
    """v2: ilk eval'in bulgularına göre (el kitabı Bölüm 10).
    - "belgede yok" bilgisi belge katmanına giriyordu      → document açıklaması yasaklıyor
    - tahminler ("muhtemelen yavaştı") çıkarıma giriyordu   → inference açıklaması yasaklıyor
    - tartışmalı konular tek taraflı anlatılıyordu         → ayrı 'debated' alanı
    - kayıt kararı modele bırakılınca tersine çalıştı      → coverage + on_topic alanları; kaydı KOD açar
    - canlı araç gelince belgeyi hiç anmadan web'den cevapladı → zorunlu 'document_says' alanı:
      model web değerlerini yazmadan ÖNCE belgede ne olduğunu söylemek zorunda (alan sırası önemli)
    Alan sırası: kanıt katmanları önce, karar alanları (coverage) sonra."""
    on_topic: bool = Field(description="True if the question is about dinosaurs, prehistoric life "
                                       "or paleontology")
    corrections: list[str] = Field(description="If the question assumes something the sources "
                                               "contradict, state the correction here and ONLY here")
    document_says: str = Field("", description="ONE sentence: what the DOCUMENTS themselves (search_docs, "
                                           "guide table) say about the main question, ignoring live "
                                           "Wikipedia and reference values. E.g. 'The documents only call "
                                           "Ceratosaurus moderately large and give no length.' If they say "
                                           "nothing: 'The documents do not cover this.'")
    document: list[Claim] = Field(description="ONLY facts the sources actually state, each with its "
                                              "source. Never write here that something is missing, "
                                              "unknown or not mentioned: that goes to not_in_document")
    inference: list[Claim] = Field(description="Conclusions that follow directly from the document "
                                               "facts above. No guesses, no 'likely/probably' "
                                               "speculation, no outside knowledge")
    web_facts: list[Claim] = Field(default_factory=list,
                                   description="Facts that came from LIVE Wikipedia or everyday reference "
                                               "objects (sources web:... or ref:...). Never put them in "
                                               "'document'. Empty if none")
    debated: list[str] = Field(description="Points the sources describe as debated, criticised, "
                                           "revised or uncertain, keeping both sides. Empty if none")
    not_in_document: list[str] = Field(description="Parts of the question the sources do not cover")
    coverage: Literal["full", "partial", "none"] = Field(
        description="Do the sources answer the MAIN question? full = yes; partial = only part of it; "
                    "none = the specific thing asked is not stated (related facts do not count)")
    general_knowledge: list[GeneralFact] = Field(description="Widely known facts about the uncovered "
                                                             "parts, NOT from the sources. Empty if not useful.")
    summary: str = Field(description="2-3 sentences in the language of the user's ORIGINAL message (Turkish if they wrote Turkish). If coverage is "
                                     "'none' or 'partial', or web_facts is not empty, the FIRST sentence "
                                     "says what the documents do not contain, and values from web_facts "
                                     "are introduced as 'according to Wikipedia'. Then the document-based answer, keeping debated "
                                     "points as debated. If general knowledge is used, end with a "
                                     "sentence starting 'Belge dışı bilgi:' or 'Outside the document:'.")


Verdict = Literal["supported", "refuted", "disputed", "not_in_document"]


class Evidence(BaseModel):
    source: str = Field(description="Chunk id exactly as shown in brackets (e.g. tyrannosaurus:72), or 'table' or 'timeline'")
    quote: str = Field(description="Words copied EXACTLY from that source, max 30 words")
    stance: Literal["supports", "refutes", "neutral"]


class FactCheck(BaseModel):
    restated_claim: str = Field(description="The claim as one precise English sentence")
    evidence: list[Evidence] = Field(description="Written BEFORE the verdict. Empty if the "
                                                 "sources say nothing about the claim.")
    verdict: Verdict = Field(description="supported = the sources state it; refuted = the sources "
                                         "contradict it (including outdated ideas the guide says "
                                         "were revised); disputed = the sources present the "
                                         "question as open or studies disagree; not_in_document "
                                         "= the sources do not cover it")
    correction: Optional[str] = Field(description="Only if refuted: the correct statement "
                                                  "according to the sources, otherwise null")
    explanation: str = Field(description="1-2 sentences in the language of the claim")


# ==========================================================================================
# 2) API şemaları
# ==========================================================================================
class Usage(BaseModel):
    calls: int
    in_tok: int
    out_tok: int
    usd: float
    latency_ms: int


class TraceStep(BaseModel):
    step: int
    tool: str
    args: dict
    result_preview: str
    error: bool


class Turn(BaseModel):
    """Sohbet geçmişinin bir turu: kullanıcının sorusu + asistanın özet cevabı."""
    question: str = Field(max_length=500)
    answer: str = Field(max_length=2000)


class Standalone(BaseModel):
    standalone_question: str = Field(description="The last message rewritten so it can be understood "
                                                 "without the conversation, in the user's language")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500, examples=["How long did T. rex live?"])
    history: list[Turn] = Field(default_factory=list, max_length=20,
                                description="Önceki turlar (eskiden yeniye). Boşsa tek soruluk istek.")
    session_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_-]{8,64}$",
                                      description="Sohbet oturumu. Varsa sunucu cevapları hatırlar ve "
                                                  "kullanıcı isterse PDF/Word/Excel dosyası üretilebilir.")
    debug: bool = Field(False, description="True ise araç adımları (trace) da döner")


class SourceInfo(BaseModel):
    """Arayüzün kaynak etiketini tıklanabilir yapması için: hangi dosyanın hangi sayfası."""
    doc: str
    section: str
    file: Optional[str] = None
    page: Optional[int] = None
    url: Optional[str] = None        # canlı Wikipedia / referans kaynağı


class ExportFile(BaseModel):
    name: str
    url: str
    format: Literal["pdf", "docx", "xlsx"]
    turns: int                                    # dosyaya giren cevap / tablo satırı sayısı
    kind: Literal["conversation", "table"] = "conversation"


class TimelineStep(BaseModel):
    t: float                                      # isteğin başından beri geçen saniye
    text: str


class AskResponse(BaseModel):
    answer: LayeredAnswerV2 | LayeredAnswer
    sources: dict[str, SourceInfo] = {}
    tickets_opened: list[str]
    tools_used: list[str]
    warnings: list[str]
    steps: int
    trace: Optional[list[TraceStep]] = None
    standalone_question: Optional[str] = None     # takip sorusu yeniden yazıldıysa: aranan soru
    files: list[ExportFile] = []                  # kullanıcı dosya istediyse: indirme linkleri
    timeline: list[TimelineStep] = []             # adım adım süre dağılımı
    cached: bool = False                          # cevap önbellekten geldi (LLM çağrısı yok)
    usage: Usage


class CheckRequest(BaseModel):
    claim: str = Field(min_length=5, max_length=500,
                       examples=["T. rex could run at 70 km/h."])


class EvidenceOut(Evidence):
    verified: bool = Field(description="Alıntı kaynak metinde gerçekten geçiyor mu? (kodla kontrol)")


class FactCheckOut(FactCheck):
    evidence: list[EvidenceOut]


class CheckResponse(BaseModel):
    result: FactCheckOut
    reliable: bool = Field(description="Karar en az bir doğrulanmış alıntıyla destekleniyor mu?")
    retrieved: list[str]
    warnings: list[str]
    usage: Usage


class Ticket(BaseModel):
    id: str
    created: str
    question: str
    missing_topic: str
    priority: Literal["low", "medium", "high"]
    status: str
