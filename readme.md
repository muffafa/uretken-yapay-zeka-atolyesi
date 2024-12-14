# Üretken Yapay Zeka Atöylesi 🤖

  <a class="header-badge" target="_blank" href="https://www.linkedin.com/in/muffafa/">
    <img alt= "Linkedin Follow" src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social">
  </a>
  <a class="header-badge" target="_blank" href="https://twitter.com/muffafa">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/muffafa?style=social">
  </a>

  <sub>Author:
    <a href="https://linktr.ee/muffafa" target="_blank">Muhammed Mustafa (muffafa) Savar</a><br>
    <small>First Commit: May, 2024</small>
    <br/>
    <small>Last Commit: Dec, 2024</small>
  </sub>

![Mustafa](<Mustafa_Header.png>)

---

**🚨 Bu içerik tamamlanmamış olup sürekli olarak güncellenecektir...**

---

## 0. 👇🏻 İçindekiler

- [Üretken Yapay Zeka Atöylesi 🤖](#üretken-yapay-zeka-atöylesi-)
  - [0. 👇🏻 İçindekiler](#0--i̇çindekiler)
  - [1. 👁️ Yapay Zeka Nedir?](#1-️-yapay-zeka-nedir)
    - [1.1 Yapay Zeka Kısa Tarihi](#11-yapay-zeka-kısa-tarihi)
    - [1.2 Yapay Zeka Türleri ve Kavramları](#12-yapay-zeka-türleri-ve-kavramları)
    - [1.3 Genel Terimler (Açıklayıcı)](#13-genel-terimler-açıklayıcı)
    - [1.4 Temel ve İleri Yapay Zeka Kavramları](#14-temel-ve-i̇leri-yapay-zeka-kavramları)
      - [1.4.1 Yapay Zeka (Artificial Intelligence)](#141-yapay-zeka-artificial-intelligence)
      - [1.4.2 Makine Öğrenmesi (Machine Learning)](#142-makine-öğrenmesi-machine-learning)
      - [1.4.3 Sinir Ağları (Neural Networks)](#143-sinir-ağları-neural-networks)
      - [1.4.4 Derin Öğrenme (Deep Learning)](#144-derin-öğrenme-deep-learning)
    - [1.5 Yapay Zekanın Kullanım Alanları](#15-yapay-zekanın-kullanım-alanları)
  - [2. 🤖 Prompt Mühendisliği ve Üretken Yapay Zeka Araçları](#2--prompt-mühendisliği-ve-üretken-yapay-zeka-araçları)
  - [3. 👀 Etik, Yasal ve Toplumsal Yönleri](#3--etik-yasal-ve-toplumsal-yönleri)
  - [4. 📒 Ek Kaynkalar](#4--ek-kaynkalar)
  - [5. 🗞️ Önemli Haberler](#5-️-önemli-haberler)
  - [6. 🎥 YouTube Kanalları](#6--youtube-kanalları)
  - [7. 🙋🏻 Diğer İçerikler](#7--diğer-i̇çerikler)

---

## 1. 👁️ Yapay Zeka Nedir?

### 1.1 Yapay Zeka Kısa Tarihi

<https://tr.wikipedia.org/wiki/Yapay_zek%C3%A2>

Yapay Zekanın Tarihsel Gelişimi

- 1956 - İstatistiksel Modelleme: İnsan zekasını taklit edebilen veya aşabilen akıllı makineler yaratmayı amaçlayan bilgisayar bilimi alanı.
  
- 1997 - Makine Öğrenmesi: Makinelerin mevcut verilerden öğrenmesini ve tahminler yapmasını sağlayan yapay zeka alt kümesi.
  
- 2017 - Derin Öğrenme: Verileri işlemek ve karar vermek için sinir ağlarının katmanlarının kullanıldığı bir makine öğrenimi tekniği.

- 2021 - Generative AI: Bilgi istemleri veya mevcut verilerle yeni yazılı, görsel ve işitsel içerik oluşturma teknolojisi.

Yapay Zeka Tarihi ve Weizenbaum Hakkında
Weizenbaum’s nightmares: how the inventor of the first chatbot turned
against AI:
<https://www.theguardian.com/technology/2023/jul/25/joseph-weizenbaum-inventor-eliza-chatbot-turned-against-artificial-intelligence-ai>

Tay Chatbot - Microsoft
<https://en.wikipedia.org/wiki/Tay_(chatbot)>

### 1.2 Yapay Zeka Türleri ve Kavramları

**Zayıf yapay zeka**, belirli görevlerde insan zekasını taklit edebilen ancak geniş çaplı zeka ve öğrenme yeteneklerine sahip olmayan sistemlerdir. Yetenekleri, tasarlandıkları görevlerle sınırlıdır. Örnekler arasında dijital asistanlar ve e-posta filtreleme sistemleri bulunur.

**Güçlü yapay zeka**, insan zekasına daha yakın yeteneklere sahip olan ve çeşitli görevlerde insan zekasını taklit edebilen sistemlerdir. Karmaşık problemlerle başa çıkabilir ve öğrenme yetenekleri sayesinde kendilerini geliştirebilirler. Dil modelleri, derin öğrenme algoritmaları, otonom araçlar ve İHA'lar bu kategoriye örnek gösterilebilir.

**Yapay genel zeka (AGI)**, insan zekasının tüm alanlarda eşdeğer veya üstün performans gösteren yapay zeka sistemleridir ve şu an için sadece teorik bir kavramdır. Tüm alanlarda genel bir anlayış ve problem çözme yeteneği sağlayabilir. Ancak, bu teknolojinin geliştirilmesi potansiyel etik ve güvenlik sorunları yaratabilir.

### 1.3 Genel Terimler (Açıklayıcı)

- **LLM (Large Language Models)**: Bir milyar veya daha fazla parametre içeren, dil işleme yeteneklerine sahip modeller. **Vektör**, dil modellerinde anlamı temsil eden sayısal değerler dizisidir.
- **Ada GPT, Dall-e2, Rainbow, Whisper, Currie**: Metin biçimlendirme, görsel oluşturma, ses komutlarını metne çevirme gibi alanlarda uzmanlaşmış modeller.
- **Fine Tuning**: Verilerin özel ihtiyaçlara göre yapay zekaya yanıt verecek şekilde uyarlanması işlemi.
- **Prompt**: Yapay zekaya ne hakkında yanıt vermesi gerektiğini belirten komut veya sorgu.
- **Token**: Yapay zeka modellerinde kelimelerin veya kelime parçalarının hesaplamalarda kullanıldığı birim. 1000 token yaklaşık 750 kelimeye eşittir. <https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them>
- **Copilot**: Kullanıcı verimliliğini artıran yapay zeka asistanı. Microsoft tarafından yayınlanmıştır. <https://copilot.microsoft.com/>
- **ChatGPT**: OpenAI tarafından geliştirilen, diyalog konusunda uzmanlaşmış yapay zeka sohbet robotu. <https://chat.openai.com/>
- **Open AI**: ChatGPT, GPT-4, GPT4-O modelini geliştiren şirket.
- **GPT (Generative Pre-Trained Transformer)**: Önceden eğitilmiş, metin üretme ve dönüştürme görevleri için kullanılan bir yapay zeka modeli. **Transformer**, veri içindeki ilişkileri öğrenmeyi sağlayan bir sinir ağı mimarisidir.
- **Llama**: Facebook tarafından geliştirilen büyük dil modeli. Açık kaynaktır. <https://llama.meta.com/>
- **Gemini**: Google tarafından geliştirilen büyük dil modeli. Google araçları ile etkileşimlidir. <https://gemini.google.com/>

### 1.4 Temel ve İleri Yapay Zeka Kavramları

![AI Concepts](ai_concepts.jpeg)

#### 1.4.1 Yapay Zeka (Artificial Intelligence)

- **Tanım**: Makinelerin insan benzeri zeka sergilemesi.
- **Kapsam**:
  - Doğal Dil İşleme (Natural Language Processing)
  - Görsel Algı (Visual Perception)
  - Otomatik Programlama (Automatic Programming)
  - Bilgi Temsili (Knowledge Representation)
  - Akıllı Robot (Intelligent Robot)
  - Otomatik Akıl Yürütme (Automatic Reasoning)

#### 1.4.2 Makine Öğrenmesi (Machine Learning)

- **Tanım**: Makinelerin verilerden öğrenmesini sağlayan yöntemler bütünü.
- **Yöntemler**:
  - Doğrusal/Logistik Regresyon (Linear/Logistic Regression)
  - K-ortalama Kümeleme (k-Means Clustering)
  - Destek Vektör Makineleri (Support Vector Machines)
  - Ana Bileşen Analizi (Principal Component Analysis)
  - K-en Yakın Komşu (k-Nearest Neighbor)
  - Rastgele Ormanlar (Random Forests)
  - Karar Ağaçları (Decision Trees)

#### 1.4.3 Sinir Ağları (Neural Networks)

- **Tanım**: İnsan beynindeki sinir hücrelerini taklit eden bilgi işleme sistemleri.
- **Yöntemler**:
  - Çok Katmanlı Algılayıcılar (MLP - Multilayer Perceptrons)
  - Boltzmann Sinir Ağları (Boltzmann Neural Networks)

#### 1.4.4 Derin Öğrenme (Deep Learning)

- **Tanım**: Verileri çok katmanlı sinir ağlarıyla işleyen makine öğrenimi yöntemi.
- **Yöntemler**:
  - Konvolüsyonel Sinir Ağları (CNN - Convolutional Neural Networks)
  - Tekrarlayan Sinir Ağları (RNN - Recurrent Neural Networks)
  - Üretici Çekişmeli Ağlar (GAN - Generative Adversarial Networks)
  - Derin İnanç Ağları (DBN - Deep Belief Networks)

### 1.5 Yapay Zekanın Kullanım Alanları

---

## 2. 🤖 Prompt Mühendisliği ve Üretken Yapay Zeka Araçları

- Prompt Yazım Teknikleri <https://github.com/atilsamancioglu/PromptEngineeringCourse/blob/main/TurkishGuide.md>
![ai framework](ai-framework.jpeg)
- Metin için Üretken Yapay Zeka Araçları: ChatGPT, Gemini
- Görsel için Üretken Yapay Zeka Araçları: Microsoft Copilot, Canva, Midjourney, Adobe Firefly
- Video için Üretken Yapay Zeka Araçları: Runway
- Ses için Üretken Yapay Zeka Araçları: Murf.AI
- Sunum için Üretken Yapay Zeka Araçları: Gamma
- Kod için Üretken Yapay Zeka Araçları: GitHub Copilot, Devin
- Diğer Üretken Yapay Zeka Araçları: ChatPDF, Notion.AI, Merlin

---

## 3. 👀 Etik, Yasal ve Toplumsal Yönleri

- Telif Hakları ve Fikri Mülkiyet
- Özgünlük ve Yaratıcılık Üzerine Etkileri
- İşgücü ve Meslekler Üzerine Etkileri
- Önyargı ve Ayrımcılık
- Yanıltıcı İçerik ve Sahte Bilgi Üretimi (Deepfake ve Dezenformasyon)
- Üretken yapay zekanın sorumlu kullanımı

---

## 4. 📒 Ek Kaynkalar

- Elements of AI: <https://course.elementsofai.com/>
- GPT-4o vs. GPT-4 vs. Gemini 1.5 — Performance Analysis <https://medium.com/@lars.chr.wiik/gpt-4o-vs-gpt-4-vs-gemini-1-5-performance-analysis-6bd207a2c580>
- [ChatGPT Her Şeyi Nasıl Bilebiliyor? (Youtube / Evrim Ağacı)](https://youtu.be/7foCbOktTZM?si=XR5aBkWGWALC5ysD)

---

## 5. 🗞️ Önemli Haberler

- Devin AI <https://www.ntv.com.tr/teknoloji/dunyanin-ilk-yapay-zeka-yazilim-muhendisi-devin,VRZ1WLa4qE2_si80pHwRXw>

---

## 6. 🎥 YouTube Kanalları

- [@cicekileteknoloji](https://www.youtube.com/@cicekileteknoloji)
- [@erhanmeydan](https://www.youtube.com/@erhanmeydan)
- [@bo-sar](https://www.youtube.com/@bo-sar)
- [@engineerprompt](https://www.youtube.com/@engineerprompt)
- [@AIJasonZ](https://www.youtube.com/@AIJasonZ)

---

## 7. 🙋🏻 Diğer İçerikler

- Grok <https://youtu.be/EdjGnqM1AjI?si=TfTw_DPJ0PDudk7Q>
- Microsoft Build 2024: Everything Revealed in 9 Minutes <https://www.youtube.com/watch?v=w1EepB0mCbE>
- Sora - Creating video from text <https://openai.com/index/sora/>
- GPT-4o Her Şeyi Değiştirir! <https://youtu.be/jZxV38uo6fE?si=Nhx9X469o9LzblHr>
