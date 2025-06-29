# Modular Hack Weekend - Hackathon Context

## Event Overview

**Event**: Modular Hack Weekend  
**Dates**: Friday, June 27th through Sunday, June 29th, 2025  
**Focus**: Mojo kernels and MAX Graph model architectures  
**Sponsor**: NVIDIA (providing GPU prizes)  
**Infrastructure**: Lambda.ai providing $400 GPU credits per participant  

## Rules & Eligibility

### Team Structure
- Teams may include up to 3 participants maximum
- Prizes awarded per team (team chooses which member receives GPU prize)
- Solo participation allowed

### Geographic Restrictions
- Due to export restrictions, prizes limited to participants in North America only

### Submission Requirements
All submissions must include:
1. Link to public GitHub repository with all source code
2. Names and email addresses of all team members
3. Link to Modular forum post describing the project

### Technical Requirements
- Submissions may include Mojo kernels OR MAX Graph architectures
- Projects must be built during hackathon period (June 27-29, 2025)
- Participants responsible for stopping Lambda GPU instances to avoid charges beyond $400 credits

### Legal
- By registering, participants agree Modular may share email with NVIDIA

## Suggested Project Categories

### Mojo Kernels (Preferred Implementations)
- **Batched matrix multiplication (BMM)**
- **Multi-head latent attention (MLA)** - Currently trending due to DeepSeek success
- **Mixture of experts (MOE)**
- **Non-maximum suppression (NMS)**
- **Grouped matrix multiplication**
- **2D convolutions**
- **General matrix-vector multiply (GEMV) on Hopper**

### MAX Graph Model Architectures (Preferred Implementations)
- **Whisper** - Audio transcription model
- **YOLO models** (such as YOLOv10) - Object detection
- **SAM or MobileSAM** - Segment Anything models
- **Bagel-7B** - Language model
- **Generative Recommenders** - Recommendation systems
- **Text or image diffusion models** (SDXL, FLUX.1/FLUX.1 Kontext)

## Preparation Resources

### GPU Programming with Mojo
- **Mojo GPU Puzzles**: Hands-on challenges to develop skills
- **Optimize custom ops for GPUs**: Focused tutorial for getting started
- **Mojo GPU documentation**: Starting point for newcomers to Mojo on GPUs

### MAX Graph Development
- **Get started with MAX graphs**: Quick guide to building MAX Graphs in Python
- **MAX graph Python API reference**: Complete API documentation

### PyTorch Integration
- **Tutorial on writing custom PyTorch kernels**: Hardware-agnostic custom ops
- **Forum post on Mojo + PyTorch support**: Community discussion and examples

## Technical Context

### Platform Capabilities
- **Mojo**: High-performance systems programming language for AI
- **MAX Graph**: Python API for building optimized model architectures
- **GPU Acceleration**: Focus on NVIDIA GPU optimization
- **Performance Focus**: Speed, memory efficiency, and hardware utilization

### Judge Preferences (Inferred)
- Practical applications with clear performance improvements
- Benchmarking against existing solutions
- Real-world use cases that demonstrate platform capabilities
- Technical innovation balanced with feasibility
- Marketing potential for Modular's platform

### Current Trends
- **Multi-Head Latent Attention (MLA)** is particularly hot due to DeepSeek's recent success
- Audio/speech processing gaining attention
- Edge deployment and efficiency optimization
- Genomics applications may be controversial (Modular removed previous content)

## Infrastructure Details

### Lambda.ai GPU Credits
- $400 in credits provided per participant
- Participants responsible for managing usage and stopping instances
- Suitable for weekend development and testing

### Development Environment
- Public GitHub repositories required
- Modular forum integration expected
- GPU-focused development workflow

## Success Factors

### Technical Excellence
- Clear performance improvements over baselines
- Proper benchmarking and measurement
- Working demonstrations
- Code quality and documentation

### Presentation
- Compelling forum post describing the project
- Clear narrative about problem solved
- Quantified results and improvements
- Potential for marketing/showcase value

### Feasibility
- Appropriate scope for weekend timeline
- Leverages team strengths and experience
- Has fallback options if full scope proves challenging
- Builds incrementally with testing at each stage