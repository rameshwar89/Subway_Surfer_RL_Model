# Subway Surfers RL Agent

This project implements an autonomous Reinforcement Learning (RL) agent that plays Subway Surfers on an Android emulator using pure computer vision and a highly optimized low-latency IPC pipeline. 

## 🎬 Demonstration

<!-- Add your training or gameplay video here -->
![Gameplay Video Placeholder](link-to-your-video.gif)

## 📈 Training Metrics (TensorBoard)

<!-- Add your TensorBoard screenshot here to show the 86+ step survival and 250k+ steps -->
![TensorBoard Metrics Placeholder](link-to-your-tensorboard-screenshot.png)

## 🚀 Key Achievements
*   **Architected** an end-to-end RL pipeline using Stable-Baselines3 (PPO), achieving an 86+ step mean survival length over 250,000+ training steps.
*   **Engineered** a low-latency Inter-Process Communication (IPC) bridge, bypassing the Android OS `adb shell` overhead by injecting raw binary touch events directly into a `scrcpy` TCP socket, dropping the send-to-input injection latency to <5ms.
*   **Optimized** real-time computer vision inference by building a threaded, non-blocking frame buffer and capping video ingestion to 30 FPS / 8 Mbps, ensuring the decision loop processed frames within a strict 16ms window.
*   **Trained** a custom YOLO26 object detection model on manually annotated gameplay datasets and integrated ByteTrack for temporal obstacle awareness, compressing raw 1080p pixel arrays into a dense 8-float state vector.
*   **Diagnosed** and resolved complex "reward hacking" failure modes across 31 training phases, mathematically balancing terminal vs. per-step penalties to eliminate boundary exploitation and episode-length inversion loops.

## 🛠 Tech Stack
*   **Python**
*   **PyTorch**
*   **Stable-Baselines3 (PPO)**
*   **YOLO26 & ByteTrack**
*   **OpenCV**
*   **scrcpy TCP Socket (Low-latency IPC)**
*   **TensorBoard**
