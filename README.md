<h1 align="center">👁️ Cataract Detection & Severity Prediction</h1>

<p align="center">
  <b>Deep Learning powered system for cataract classification and severity estimation</b><br>
  Built with <code>PyTorch</code>, <code>Torchvision</code>, and <code>ResNet</code> architectures.
</p>

<hr>

<h2>📌 Overview</h2>
<p>
This repository contains two complementary modules:
</p>
<ul>
  <li><b>main.py</b> – Trains a <code>ResNet50</code> model for <b>severity prediction</b> (regression output as percentage).</li>
  <li><b>test.py</b> – Loads a <code>ResNet18</code> model for <b>binary classification</b> (Cataract vs No Cataract) and predicts on new images.</li>
</ul>

<hr>

<h2>✨ Features</h2>
<ul>
  <li>📊 <b>Severity Prediction</b> – Outputs cataract severity as a percentage using regression (MSE loss).</li>
  <li>🩺 <b>Binary Classification</b> – Distinguishes between <i>Cataract</i> and <i>No Cataract</i> cases.</li>
  <li>⚡ <b>Transfer Learning</b> – Fine-tunes pre-trained ResNet models for medical imaging tasks.</li>
  <li>🎯 <b>Data Augmentation</b> – Includes resizing, rotation, color jitter, and normalization for robust training.</li>
  <li>💾 <b>Model Persistence</b> – Saves trained weights for later inference.</li>
</ul>

<hr>

<h2>🛠️ Tech Stack</h2>
<ul>
  <li><b>Frameworks:</b> PyTorch, Torchvision</li>
  <li><b>Models:</b> ResNet50 (regression), ResNet18 (classification)</li>
  <li><b>Tools:</b> Matplotlib, PIL</li>
</ul>

<hr>

<h2>⚙️ Usage</h2>

<h3>Training (Severity Prediction)</h3>
<pre>
python main.py
</pre>
<p>
This will:
</p>
<ul>
  <li>Train <code>ResNet50</code> on cataract images</li>
  <li>Save weights to <code>cataract_severity_model.pth</code></li>
  <li>Report mean severity prediction error</li>
</ul>

<h3>Testing (Classification)</h3>
<pre>
python test.py
</pre>
<p>
This will:
</p>
<ul>
  <li>Load <code>ResNet18</code> with trained weights (<code>cataract_model.pth</code>)</li>
  <li>Predict Cataract vs No Cataract for a given image</li>
  <li>Print the predicted class</li>
</ul>

<hr>

<h2>📂 Repository Structure</h2>
<pre>
├── main.py        # Train ResNet50 for severity regression
├── test.py        # Test ResNet18 for binary classification
├── Cataract/
│   └── processed_images/
│       ├── train/
│       └── test/
└── README.md
</pre>


