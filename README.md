# Semi-Parallel Deep Neural Network (SPDNN) Hybrid Architecture, First Application on Depth from Monocular Camera
This repository contains the code for the method presented in the following paper:

**Bazrafkan, S., Javidnia, H., Lemley, J. and Corcoran, P., 2017. "Depth from Monocular Images using a Semi-Parallel Deep Neural Network (SPDNN) Hybrid Architecture". arXiv preprint arXiv:1703.03867**



As described in the Training section of the paper, four experiments are designed in this project:

**Exp1:** Input: Left Visible Image + Pixel-wise Segmented Image. Target: Post-Processed Depth map.

**Exp2:** Input: Left Visible Image. Target: Post-Processed Depth map.

**Exp3:** Input: Left Visible Image + Pixel-wise Segmented Image. Target: Depth map.

**Exp4:** Input: Left Visible Image. Target: Depth map.



To prepare the input for training:

1- Install the [Caffe SegNet](https://github.com/alexgkendall/caffe-segnet)

2- Train the SegNet using [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) road scene database

3- Use the trained model to segment the images of the [KITTI 2012, 2015](http://www.cvlibs.net/datasets/kitti/eval_stereo.php) dataset.



To prepare the target for training:

1- Estimate the depth from the [KITTI stereo sets](http://www.cvlibs.net/datasets/kitti/eval_stereo.php) using [Adaptive Random Walk with Restart algorithm](https://www.sciencedirect.com/science/article/pii/S0262885615000104)

2- Post-process the initial depth maps using our [post-processing method](https://github.com/hosseinjavidnia/Post-Processing-ARWR)

You can duplicate the experiments described in the paper using the codes in this repository and the prepared data.


Please cite the following papers when you are using this code:

**Bazrafkan, S., Javidnia, H., Lemley, J. and Corcoran, P., 2017. "Depth from Monocular Images using a Semi-Parallel Deep Neural Network (SPDNN) Hybrid Architecture". arXiv preprint arXiv:1703.03867**
