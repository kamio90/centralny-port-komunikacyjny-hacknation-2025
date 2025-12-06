# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a HackNation2025 hackathon project for CPK (Centralny Port Komunikacyjny) focused on **automatic classification of infrastructure elements from point clouds** (LiDAR data). The goal is to develop a computational module that automatically classifies point cloud data into infrastructure categories to support BIM (Building Information Modeling) workflows.

### Challenge Context

CPK requires automated point cloud classification to:
- Monitor construction progress
- Verify quality of construction work
- Compare real-world state with 3D BIM models
- Reduce manual, time-consuming classification work

The solution must classify points into infrastructure categories such as:
- Road surfaces (nawierzchnie dróg)
- Railway tracks (torowiska)
- Curbs (krawężniki)
- Poles (słupy)
- Signs (znaki)
- Barriers (bariery)
- Vegetation (zieleń)
- Buildings (zabudowania)
- Pipelines (rurociągi)
- High voltage lines (linie wysokiego napięcia)
- Traction systems (trakcja)

## Data Files

### Available Resources
- **Reference point cloud**: `hackaton_task_files/Chmura_Zadanie_Dane.las` (7.2GB LiDAR data, version 1.2)
- **Task specification**: `hackaton_task_files/CHMURA_POD_KONTROLA.pdf`
- **Class documentation**: `hackaton_task_files/Dokumentacja klas.docx`
- **Regulations**: `hackaton_task_files/Regulamin-CHMURA_POD_KONTROLA.pdf`

**IMPORTANT**: These files are exclusively for HackNation2025 and cannot be used for other purposes.

## Technical Requirements

### Input/Output Formats
- **Input**: LAS/LAZ files (mandatory)
- **Output**:
  - LAS/LAZ files with assigned classifications (mandatory)
  - IFC files with reconstructed infrastructure elements (bonus)

### Architecture Expectations
The solution should be a **computational module**, not a full application:
- Modular design for easy integration with CDE (Common Data Environment) systems
- Simple interface (CLI or minimal web UI)
- Scalable processing for large datasets (using tiling, streaming, or batch processing)
- Deterministic and repeatable algorithm
- Docker containerization (recommended)

### Technology Constraints
- Use open-source or free libraries suitable for pilot/research projects
- Avoid proprietary SaaS-only tools that cannot be deployed in corporate environments
- Avoid tools generating closed, undocumented data formats
- Support integration with BIM analysis tools and 3D model comparison systems

### Minimum Classification Target
- Must classify at least **5 object classes**
- Algorithm can be ML-based, DL-based, or heuristic-based

## Deliverables

The final submission must include:

1. **Project title and description** (goal, assumptions, algorithm approach)
2. **PDF presentation** (max 10 slides):
   - Module architecture
   - Methods used
   - Processing approach for reference point cloud
   - Example results
3. **Video demonstration** (max 3 minutes):
   - Show solution working on the reference point cloud
   - Visualization of classification results
4. **Installation instructions**:
   - Input/output formats
   - Installation steps
   - How to run the module
5. **Processed reference point cloud output**:
   - LAS/LAZ with assigned classes
   - IFC file (bonus)

Optional but recommended:
- Code repository (GitHub/GitLab)
- Docker image
- Screenshots/graphics showing classification process
- Demo link (web UI or preview panel)

## Validation & Testing

### Two-Dataset Approach
- **Training/development**: Teams work with the provided reference LAS file
- **Evaluation**: Jury will test on a separate, unseen LAS file with similar characteristics

This approach validates the algorithm's **generalization capability**.

### Evaluation Metrics
1. **Classification accuracy** (on unseen data):
   - Number of processed points
   - Number of classified points
   - Number of unclassified points
   - Confusion matrix
   - Visual assessment

2. **Completeness and correctness**:
   - Full output file delivery (LAS/LAZ, possibly IFC)
   - Class, geometry, and metadata compliance

3. **Performance and stability**:
   - Processing time for test file
   - No errors during execution
   - Clear installation/setup process
   - Modularity and extensibility

## Evaluation Criteria (100% total)

- **Classification effectiveness**: 30%
- **Challenge compliance and technical correctness**: 20%
- **Algorithm design/approach**: 20%
- **Extensibility potential (new classes)**: 20%
- **Deployment potential**: 10%

## Point Cloud Processing Libraries

Consider using:
- **Python**: laspy, pdal, open3d, pyntcloud, CloudComPy, pylas
- **C++**: PDAL, PCL (Point Cloud Library), LAStools, CGAL
- **Machine Learning**: PyTorch, TensorFlow, scikit-learn (for ML/DL approaches)

## BIM Integration Context

The classified point clouds must enable comparison with 3D BIM models. Understanding BIM methodology is important:
- BIM = Building Information Modeling
- Digital 3D models with linked technical information
- Covers entire infrastructure lifecycle: design, construction, acceptance, maintenance
- Requires coordinated, synchronized data updates

## Future Opportunities

Top solutions may be invited for:
- Continued collaboration with CPK
- Pilot testing on real construction data
- Technical consultations with CPK Digital Transformation team
- Development into production-grade module for quality monitoring

## Contact & Support

Mentors available on Discord channels:
- #zapytaj-mentora
- #chmura-pod-kontrolą

CPK Mentors:
- Michał LATAŁA – Director, Digital Transformation Office
- Agnieszka NOWAK-DALEWSKA – Point Cloud Management Expert
- Leszek MAJKA – BIM Expert, Infrastructure Modeling Standards
