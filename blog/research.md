---
layout: research
title: Research
description: Mauro Comi's academic profile

publications:
  - img: "../img/research/survey_paper_NF.png"
    title: "Neural Fields in Robotics: A Survey"
    link: "https://arxiv.org/pdf/2410.20220v1"
    authors: "Muhammad Zubair Irshad, <strong>Mauro Comi</strong>, Yen-Chen Lin, Nick Heppert, Abhinav Valada, Rares Ambrus, Zsolt Kira, Jonathan Tremblay"
    status: "Currently in submission, available on Arxiv"
  - img: "../img/research/snap_tap.png"
    title: "Snap-it, Tap-it, Splat-it: Tactile-Informed 3D Gaussian Splatting for Reconstructing Challenging Surfaces"
    link: "https://arxiv.org/abs/2403.20275"
    authors: "<strong>Mauro Comi</strong>, Alessio Tonioni, Max Yang, Jonathan Tremblay, Valts Blukis, Yijiong Lin, Nathan F. Lepora, Laurence Aitchison"
    status: "<strong>3DV 2025</strong>"
  - img: "../img/research/results_increasing_touch.png"
    title: "TouchSDF: A DeepSDF Approach for 3D Shape Reconstruction Using Vision-Based Tactile Sensing"
    link: "https://arxiv.org/abs/2311.12602"
    authors: "<strong>Mauro Comi</strong>, Yijiong Lin, Alex Church, Alessio Tonioni, Laurence Aitchison, Nathan F. Lepora"
    status: "<strong>IEEE RA-L</strong>, also 3DVR workshop at <strong>CVPR 2023</strong>, and Touch processing workshop at <strong>NeurIPS 2023</strong>"
  - img: "../img/research/tactile_saliency.png"
    title: "Attention of Robot Touch: Tactile Saliency Prediction for Robust Sim-to-Real Tactile Control"
    link: "https://arxiv.org/pdf/2307.14510.pdf"
    authors: "Yijiong Lin, <strong>Mauro Comi</strong>, Alex Church, Dandan Zhang, Nathan F. Lepora"
    status: "<strong>IROS 2023</strong>"
  - img: "../img/research/safeai.png"
    title: "A Hybrid-AI approach to Competence Assessment for Automated Driving Functions"
    link: "http://ceur-ws.org/Vol-2808/Paper_37.pdf"
    authors: "Jan-Pieter Paardekooper, <strong>Mauro Comi</strong>, Corrado Grappiolo, Ron Snijders, Willeke van Vught, Rutger Beekelaar"
    status: "<strong>SafeAI AAAI 2021</strong>"
---
---

<img id="img-profile" src="../img/jumping_me.png" alt="A picture of me jumping in front of a lighthouse">

Hi there! I am currently a research intern at [Google DeepMind](https://deepmind.google/) and a PhD student in **Machine Learning** at the [University of Bristol](https://www.bristol.ac.uk/) (UK). My research interests lie at the intersection of **3D Deep Learning, Neural Fields and 3D Gaussian Splatting,** and **Robotics Perception** (Computer Vision, Tactile Sensing). I have a soft spot for **Computer Graphics** and **Physically-Based Rendering**. I am supervised by [Prof Nathan Lepora](https://lepora.com/) and [Dr Laurence Aitchison](http://www.gatsby.ucl.ac.uk/~laurence/), and I am fortunate to collaborate with and get guidance from [Alessio Tonioni](https://alessiotonioni.github.io/) (Google) and [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay) (NVIDIA). Previously, I worked as a Machine Learning research engineer in autonomous driving at the Netherlands Organisation for Applied Scientific Research (TNO), where I worked and led EU-funded projects on autonomous driving, and developed **Deep Reinforcement Learning** applications for self-driving vehicles.

I read papers following Andrew Ng's invaluable tips on <a href="https://youtu.be/733m6qBH-jI">How to read research papers (Andrew NG)</a>

### Publications
<div style="display: flex; flex-direction: column; gap: 20px;">
{% for publication in page.publications %}
    <div class="publication" style="display: flex; align-items: center; gap: 20px; border-bottom: 1px solid #ddd; padding-bottom: 15px;">
        <img src="{{ publication.img }}" alt="{{ publication.title }}" width="200px" style="border-radius: 10px;">
        <div>
            <a href="{{ publication.link }}" style="font-size: 1.1em; font-weight: bold;">{{ publication.title }}</a><br>
            <span style="color: #555;">{{ publication.authors }}</span><br>
            <span style="font-size: 0.9em; font-weight: bold; color: #0073e6;">{{ publication.status }}</span>
        </div>
    </div>
{% endfor %}
</div>

### Reading group

I run an online **3D Deep Learning reading group**, where we discuss papers in 3D Vision for the virtual and physical world. If you are interested or want to join, please visit the [reading group website](https://3d-deeplearning-rg.github.io/).  

### Teaching
- **[Introduction to AI](https://www.bris.ac.uk/unit-programme-catalogue/UnitDetails.jsa?ayrCode=22%2F23&unitCode=EMATM0044), Teaching Assistant**, BSc unit, MSc unit, @University of Bristol, 2021/2022, 2021/2023
- **Anomaly Detection using Machine Learning, Guest Lecturer** @Jheronimus Academy of Data Science, July 2021


### Talks
- **A Hybrid-AI approach to Competence Assessment for Automated Driving Functions** @SafeAI AAAI, February 2021

### Updates
- **February 2025** ⭐ – I started my second research internship as a **Student Researcher** at **Google DeepMind**!  
- **November 2024** – Our work *Snap-it, Tap-it, Splat-it: Tactile-Informed 3D Gaussian Splatting for Reconstructing Challenging Surfaces* ([link](https://arxiv.org/pdf/2410.20220v1)) was accepted at [3DV 2025](https://3dvconf.github.io/2025/). See you in Singapore!  
- **October 2024** – Our work *Neural Fields in Robotics: A Survey* was submitted to ArXiv ([link](https://arxiv.org/pdf/2410.20220v1)).  
- **September 2024** ⭐ – I started my research internship as a **Student Researcher** at **Google DeepMind**!  
- **May 2024** - Our work *TouchSDF: A DeepSDF Approach for 3D Shape Reconstruction Using Vision-Based Tactile Sensing* was accepted into the [IEEE RA-L journal](https://ieeexplore.ieee.org/abstract/document/10517361).
- **April 2024** - The 3D Deep Learning Reading Group I am running is starting again. Join us [here](https://3d-deeplearning-rg.github.io/).
- **October 2023** - Our work *TouchSDF: A DeepSDF Approach for 3D Shape Reconstruction Using Vision-Based Tactile Sensing* was accepted to the Touch processing in AI workshop at **NeurIPS 2024**. See you in New Orleans!
- **May 2023** - Our work *TouchSDF: A DeepSDF Approach for 3D Shape Reconstruction Using Vision-Based Tactile Sensing* was accepted to the 3D Vision and Robotics workshop at **CVPR 2023**. See you in Vancouver!
- **April 2023** - Our work *Implicit Neural Representation for 3D Shape Reconstruction Using Vision-Based Tactile Sensing* was accepted to the workshop ViTac: Blending Virtual and Real Visuo-Tactile Perception at **ICRA 2023**. See you in London!
- **September 2021** - I started my PhD in Machine Learning at the University of Bristol.
- **February 2021** - Our work *A Hybrid-AI approach to Competence Assessment for Automated Driving Functions* was accepted at **SafeAI AAAI 2021**.
- **November 2019** - I graduated Cum Laude in Data Science with a MSc thesis on Deep RL for physically-based rendering.
- **April 2019** - I started to work as Machine Learning research engineer at the Netherlands Organisation for Applied Scientific Research (TNO).

---