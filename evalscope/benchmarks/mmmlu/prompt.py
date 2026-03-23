# Copyright (c) Alibaba, Inc. and its affiliates.
# Per-language translations of SINGLE_ANSWER_TEMPLATE_COT for MMMLU.
# All templates keep 'ANSWER: [LETTER]' as the response format so that
# the standard parse_answers() extractor works across all languages.

# flake8: noqa: E501

AR_XY = r"""أجب على سؤال الاختيار من متعدد التالي. يجب أن يكون السطر الأخير من إجابتك بالتنسيق التالي: 'ANSWER: [LETTER]' (بدون علامات اقتباس) حيث [LETTER] هو أحد الحروف {letters}. فكّر خطوة بخطوة قبل الإجابة.

{question}

{choices}""".strip()

BN_BD = r"""নিচের বহুনির্বাচনী প্রশ্নের উত্তর দিন। আপনার উত্তরের শেষ লাইনটি এই ফরম্যাটে হওয়া উচিত: 'ANSWER: [LETTER]' (উদ্ধৃতি চিহ্ন ছাড়া) যেখানে [LETTER] হল {letters}-এর মধ্যে একটি। উত্তর দেওয়ার আগে ধাপে ধাপে চিন্তা করুন।

{question}

{choices}""".strip()

DE_DE = r"""Beantworte die folgende Multiple-Choice-Frage. Die letzte Zeile deiner Antwort sollte folgendes Format haben: 'ANSWER: [LETTER]' (ohne Anführungszeichen), wobei [LETTER] eines der folgenden Zeichen ist: {letters}. Denke Schritt für Schritt, bevor du antwortest.

{question}

{choices}""".strip()

ES_LA = r"""Responde la siguiente pregunta de opción múltiple. La última línea de tu respuesta debe tener el siguiente formato: 'ANSWER: [LETTER]' (sin comillas) donde [LETTER] es una de las opciones {letters}. Piensa paso a paso antes de responder.

{question}

{choices}""".strip()

FR_FR = r"""Répondez à la question à choix multiple suivante. La dernière ligne de votre réponse doit être au format suivant : 'ANSWER: [LETTER]' (sans guillemets) où [LETTER] est l'une des options {letters}. Réfléchissez étape par étape avant de répondre.

{question}

{choices}""".strip()

HI_IN = r"""निम्नलिखित बहुविकल्पीय प्रश्न का उत्तर दें। आपके उत्तर की अंतिम पंक्ति इस प्रारूप में होनी चाहिए: 'ANSWER: [LETTER]' (उद्धरण चिह्नों के बिना) जहाँ [LETTER] {letters} में से एक है। उत्तर देने से पहले चरणबद्ध तरीके से सोचें।

{question}

{choices}""".strip()

ID_ID = r"""Jawablah pertanyaan pilihan ganda berikut. Baris terakhir jawaban Anda harus dalam format berikut: 'ANSWER: [LETTER]' (tanpa tanda kutip) di mana [LETTER] adalah salah satu dari {letters}. Pikirkan langkah demi langkah sebelum menjawab.

{question}

{choices}""".strip()

IT_IT = r"""Rispondi alla seguente domanda a scelta multipla. L'ultima riga della tua risposta deve essere nel seguente formato: 'ANSWER: [LETTER]' (senza virgolette) dove [LETTER] è una delle opzioni {letters}. Pensa passo dopo passo prima di rispondere.

{question}

{choices}""".strip()

JA_JP = r"""次の多肢選択問題に答えてください。回答の最終行は次の形式にしてください：'ANSWER: [LETTER]'（引用符なし）。ここで [LETTER] は {letters} のいずれかです。回答する前に、ステップごとに考えてください。

{question}

{choices}""".strip()

KO_KR = r"""다음 객관식 문제에 답하세요. 응답의 마지막 줄은 다음 형식이어야 합니다: 'ANSWER: [LETTER]' (따옴표 없이) 여기서 [LETTER]는 {letters} 중 하나입니다. 답하기 전에 단계별로 생각하세요.

{question}

{choices}""".strip()

PT_BR = r"""Responda à seguinte questão de múltipla escolha. A última linha da sua resposta deve estar no seguinte formato: 'ANSWER: [LETTER]' (sem aspas) onde [LETTER] é uma das opções {letters}. Pense passo a passo antes de responder.

{question}

{choices}""".strip()

SW_KE = r"""Jibu swali lifuatalo la chaguo nyingi. Mstari wa mwisho wa jibu lako unapaswa kuwa katika muundo ufuatao: 'ANSWER: [LETTER]' (bila alama za nukuu) ambapo [LETTER] ni moja ya {letters}. Fikiria hatua kwa hatua kabla ya kujibu.

{question}

{choices}""".strip()

YO_NG = r"""Dahun ibeere yiyan-pupọ atẹle yii. Ila ikẹhin ti idahun rẹ yẹ ki o wa ni ọna atẹle yii: 'ANSWER: [LETTER]' (laisi awọn ami ifọwọsi) nibiti [LETTER] jẹ ọkan ninu {letters}. Ronu igbese ni igbese ṣaaju ki o to dahun.

{question}

{choices}""".strip()

ZH_CN = r"""回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式：'ANSWER: [LETTER]'（不带引号），其中 [LETTER] 是 {letters} 中的一个。请在回答前进行一步步思考。

{question}

{choices}""".strip()

# Maps language subset code -> prompt template string
LANGUAGE_PROMPT_MAP = {
    'AR_XY': AR_XY,
    'BN_BD': BN_BD,
    'DE_DE': DE_DE,
    'ES_LA': ES_LA,
    'FR_FR': FR_FR,
    'HI_IN': HI_IN,
    'ID_ID': ID_ID,
    'IT_IT': IT_IT,
    'JA_JP': JA_JP,
    'KO_KR': KO_KR,
    'PT_BR': PT_BR,
    'SW_KE': SW_KE,
    'YO_NG': YO_NG,
    'ZH_CN': ZH_CN,
}
