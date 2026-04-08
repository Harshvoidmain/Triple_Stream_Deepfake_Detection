# 🛡️ DeepDetect Sentinel (Dockerized)

This repository contains the full **TripleStream Deepfake Detection Pipeline** deployed into a robust, high-performance Docker container. 

The backend runs a complex multi-stream PyTorch model (DeiT-Small spatial attention, Transformer temporal encoder, and CNN-based frequency analyzer) with inference acceleration enabled by Nvidia CUDA.

## 🚀 Deployment Instructions

### Prerequisites
1. **Docker Engine** installed on your server or machine.
2. *(Optional but Highly Recommended)* **NVIDIA Container Toolkit** installed to pass your GPU through to the Docker container for hardware-accelerated video deepfake analysis!

### Step 1: Clone the build

Pull the code and `cd` into the project directory.

### Step 2: Build and Run

Bring the infrastructure up in detached mode using Docker Compose:

```bash
docker compose up --build -d
```

*Note: Since the backend is running PyTorch, the base image download will be approximately ~4GB the very first time you pull. Go grab a coffee!*

### Step 3: Access the Interface

Once `docker-compose` finishes provisioning and boot completes, access the application directly from any browser on your network: 
👉 **http://localhost:5000** 

## 🛠️ Managing with `lazydocker`

If you have `lazydocker` installed, you can simply spin it up in this directory to manage the container:

```bash
lazydocker
```

From lazydocker you can:
- **Monitor the Healthcheck:** Watch the custom Flask API `/api/health` status.
- **View Inference Logs:** See frame extraction speed and model layer metrics streaming in the application logs tab.
- **Check Resource Usage:** Keep an eye on system RAM and container memory limits.

## Troubleshooting

- **"Failed to initialize NVML" or "No GPU available" in docker logs**: 
   If you don't have an Nvidia GPU installed, or the Nvidia Toolkit isn't enabled in docker, edit the `docker-compose.yml` to remove the `deploy` block defining GPU reservations. The application will then automatically backfall to CPU compute.
   
- **"Port 5000 is already in use"**: 
  If something else on your server binds to port 5000, modify the `ports:` mapping in `docker-compose.yml` from `"5000:5000"` to your preferred port mapping (e.g. `"8080:5000"`).

---
*Built with ❤️ utilizing the Forensic Lens v3.4 UI framework.*
