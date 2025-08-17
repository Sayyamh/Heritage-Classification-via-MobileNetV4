Optimization of MobileNetV4 for Heritage Site Classification

This project focuses on efficient image classification of Indian heritage sites using the MobileNetV4 architecture with Post-Training Quantization (PTQ) for deployment on resource-constrained edge devices.

📌 Overview

Deployed MobileNetV4 (MNv4-Conv-M) for classification on the Indian Heritage Digital Space (IHDS) dataset containing 133k+ images across 50 classes.

Compared performance with MobileNetV3 under different quantization levels (4-bit, 6-bit).

Implemented custom PTQ to compress Conv2D layers while retaining high accuracy.

🚀 Key Results

MobileNetV4 original size: 33.99 MB → 5.38 MB (6-bit) / 4.05 MB (4-bit)

Accuracy after PTQ: 91.47% (6-bit), 88.62% (4-bit)

Outperforms MobileNetV3, which dropped to 87.50% (6-bit) and 82.30% (4-bit).

Provides a balance of compact size + robust accuracy ideal for edge deployment.

⚙️ Features

Lightweight MobileNetV4 with Universal Inverted Bottleneck (UIB) and Mobile Multi-Query Attention (MQA).

PTQ-based model compression for efficient deployment.

Web-based demo (Flask app) for uploading and classifying heritage site images.

🌍 Applications

Cultural heritage documentation & restoration.

Real-time image classification on mobile and edge devices.

Extendable to medical imaging, autonomous vehicles, and IoT applications.
