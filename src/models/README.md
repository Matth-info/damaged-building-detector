Available Models
This project provides implementations of several deep learning models for semantic segmentation and change detection, including Siamese architectures for bi-temporal and multi-temporal analysis. Most models are implemented directly in the models/ folder, except for SegFormer, which is a PyTorch wrapper around the HuggingFace Transformers version (and also MaskRCNN which a multi-target trained torchvision model.)

Several models rely on foundation encoders tailored for remote sensing applications.

Implemented Models
Below is the list of models included in the project, along with their corresponding research papers and reference implementations:

| Model Name                        | Research Paper                                                                                               | Reference Implementation                                    |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| **Bi-temporal Transformer (BiT)** | [BiT: A Deep Learning Framework for Remote Sensing Image Change Detection](https://arxiv.org/pdf/2103.00208) | [GitHub](https://github.com/justchenhao/BIT_CD/tree/master) |
| **ChangeFormer**                  | [ChangeFormer: A Transformer-Based Change Detection Method](https://arxiv.org/abs/2201.01293)                | [GitHub](https://github.com/wgcban/ChangeFormer)            |
| **TinyCD**                        | [TinyCD: Lightweight Change Detection with Lightweight Backbones](https://arxiv.org/abs/2207.13159)          | [GitHub](https://github.com/AndreaCodegoni/Tiny_model_4_CD) |
| **UNet**                          | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)          | [GitHub](https://github.com/milesial/Pytorch-UNet)          |
| **MaskRCCN**                      | [Mask R-CNN](https://arxiv.org/abs/1703.06870)                                                               | [Pytorch Doc](https://docs.pytorch.org/vision/main/models/mask_rcnn.html) |
| **SegFormer**                     | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | [GitHub](https://github.com/NVlabs/SegFormer)     |
| **Siam-Unet**                     | Not Found                                                                                                    |   Not Found                                                 |
| **PrithviSeg**                    | [Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications](https://arxiv.org/abs/2412.02732) | [HuggingFace Hub](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)  |
| **ClaySeg**                       | [Clay Foundation Model, An open source AI model for Earth](https://clay-foundation.github.io/model/index.html) | [GiHub](https://github.com/Clay-foundation/model) |
| **ScaleMAESeg** | [Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning](https://arxiv.org/abs/2212.14532) | [GitHub](https://github.com/bair-climate-initiative/scale-mae) |
