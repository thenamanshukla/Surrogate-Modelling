# IIT Madras Internship - 2026
## Gaussian Processes for Surrogate Modelling

<img src="./img/gp_regression.gif" width="1000"/>

*An original Gaussian Process regression animation generated for this repository. Starting from the prior, each new observation is revealed one at a time: the posterior mean (blue) bends to pass through the data, the shaded band showing the model's uncertainty collapses at the observed points and stays wide where data is absent, and the faint green curves are posterior function samples that agree closely where the model is confident and fan out where it is not. The dashed pink curve is the true function being learnt. This picture of a distribution over functions that tightens as it learns is the single idea that ties every project in this repository together.*

Summer Research Internship, Department of Mathematics, Centre of Excellence for Data Science and Computational Mathematics, Indian Institute of Technology Madras, 2026.

Carried out by Naman Shukla (Department of Computer Science and Engineering, National Institute of Technology Mizoram) under the supervision of Prof. Neelesh Shankar Upadhye, Department of Mathematics, IIT Madras.

## Overview

This repository holds the complete body of work I produced during my summer research internship at the Centre of Excellence for Data Science and Computational Mathematics, IIT Madras. My core work centered on the first four chapters of Rasmussen and Williams, Gaussian Processes for Machine Learning, covering regression, classification, and the theory of covariance functions, and on translating that theory into applied surrogate models. The theme running through every project is surrogate modelling with Gaussian Processes: building probabilistic models that not only predict, but also report how confident they are about each prediction. Over the course of the internship I moved from the foundational theory of Gaussian Processes through to applied surrogates spanning finance, computational hardware, macroeconomics, engineering simulation, and computer vision.

## A Note on the Journey

I want to be honest about how this started. When I began, the mathematics behind Gaussian Processes felt genuinely intimidating. The idea of a distribution over functions, kernels acting as inner products in infinite dimensional spaces, Mercer's theorem, Bochner's theorem, the marginal likelihood balancing fit against complexity: none of it was intuitive to me at first, and the early weeks were slow and often discouraging. Concepts that now feel natural took real effort to internalise, and there were points where the gap between the theory in the textbook and a working model on real data seemed very wide.

What changed was time and persistence. Working through the derivations carefully, preparing the two theory presentations in this repository, and then forcing myself to implement each idea against real datasets gradually turned that intimidation into understanding. Each project taught me something the previous one had left unanswered, and by the end I was comfortable reasoning about kernel choice, approximate inference, and uncertainty decomposition in settings I would not have attempted at the start. The progression in this repository reflects that arc, from theory to increasingly ambitious applications.

## Acknowledgements

I am deeply grateful to Prof. Neelesh Shankar Upadhye of the Department of Mathematics, IIT Madras, for his guidance, patience, and encouragement throughout this internship. His supervision turned a subject I initially found daunting into one I can now work in with confidence, and the direction he provided shaped every project in this repository.

I also thank the Department of Mathematics and the Centre of Excellence for Data Science and Computational Mathematics at the Indian Institute of Technology Madras for hosting me and for providing the environment and resources that made this work possible. This internship has been a formative experience in my development as a researcher, and I am thankful for the opportunity.

## What I Learnt

Across the internship I built a working command of the following areas:

The theoretical foundations of Gaussian Processes, including the function space view, the role of the mean and covariance functions, consistency and the Kolmogorov extension theorem, prediction in the noise free and noisy settings, the log marginal likelihood as a balance between data fit and model complexity, and the use of the Cholesky decomposition for numerically stable inference.

The theory of covariance functions, including Mercer's theorem and the eigenfunction view of kernels, Bochner's theorem for stationary kernels, eigenvalue decay as a measure of smoothness, the Nystrom method for numerical approximation of eigenfunctions, piecewise polynomial kernels with compact support, non stationary constructions such as the Gibbs and Paciorek Schervish kernels, and the rules for building new valid kernels from existing ones through sums, products, scaling, convolution, and combinations across input spaces.

Non stationary kernels, meaning covariance functions whose behaviour depends on absolute position rather than only on the distance between points, understood through Brownian motion as the canonical reference example. The Wiener process, with its minimum covariance function, is the fundamental non stationary kernel, and working through it made the wider family clear, including varying length scale constructions such as the Gibbs and Paciorek Schervish kernels. Brownian motion also gave a concrete picture of the Markov structure that shows up as a sparse precision matrix, in contrast to the dense precision of a stationary kernel like the squared exponential.

Approximate inference for non conjugate likelihoods, including the Laplace approximation and expectation propagation implemented from first principles, and sparse variational inference for scaling Gaussian Processes to large datasets.

Deep kernel learning, where a neural network feature extractor is trained jointly with a Gaussian Process head, along with the practical failure modes such as posterior collapse and how to prevent them through constrained hyperparameters, sensible inducing point initialisation, and balanced optimisation.

Gaussian Process classification in depth, understanding why it has no closed form posterior unlike regression, since the class label is tied to the latent function through a non Gaussian likelihood such as the probit, and working through the two classic approximate inference schemes for this setting, the Laplace approximation built around a Newton search for the posterior mode and expectation propagation built around iterative moment matching of local site approximations. Implementing both from scratch, rather than calling a library, was where the theory finally became concrete for me, and it taught me why the two methods agree on hard decisions yet differ in how well their probabilities are calibrated.

Multi fidelity modelling, uncertainty propagation through nonlinear maps, and honest model evaluation using held out predictive scoring rather than in sample fit.

Much of this grounding came from the classic reference in the field, Gaussian Processes for Machine Learning by Carl Edward Rasmussen and Christopher K. I. Williams (MIT Press, 2006), which the authors and MIT Press have generously made freely available online at [gaussianprocess.org/gpml](https://gaussianprocess.org/gpml/). Several projects in this repository, particularly the classification work, follow its chapters and algorithms directly, and I would recommend it to anyone starting out.

If I had to name a favourite, it would be Bochner's theorem and the eigenvalue decomposition view of kernels. There is something that genuinely stayed with me in seeing a covariance function unfold into a spectrum of frequencies, and in understanding a kernel as an orthogonal basis with eigenvalues that measure how much a process varies along each direction, exactly mirroring how a covariance matrix diagonalises in finite dimensions. That the smoothness of a process is encoded in the decay rate of its eigenvalues is, to me, one of the most elegant ideas I met during the internship.

Toward the end, Prof. Neelesh Shankar Upadhye put the next challenge plainly: now that I knew these ideas individually, the real work was to connect them, to see the classification approximations, the spectral view of kernels, the multi fidelity constructions, and the scalable approximations not as separate topics but as one connected picture. That framing has shaped how I now approach the subject and is the direction I intend to carry forward.

## Future Work

A recurring lesson across the internship was that going deep in any one direction required heavy prior knowledge specific to that area, whether that was financial time series structure for the crypto work, GPU memory and tiling behaviour for the hardware surrogates, or the measure theory underlying the spectral view of kernels. The natural next step is to build that specialised depth in the directions I found most compelling, and to act on Prof. Upadhye's guidance by connecting these threads rather than treating them separately: extending the spectral and eigenfunction view toward non stationary kernels in high dimensions, tying the approximate inference schemes to the scalable variational methods, and grounding each applied surrogate in the domain knowledge its problem demands.

## Theory Presentations

The theoretical grounding for the internship is captured in two presentations, available as PDFs in the `presentations` folder.

`FunctionSpaceView.pdf` develops the function space view of Gaussian Processes: the kernel trick, the formal definition of a Gaussian Process, why the framework works well on small datasets, consistency and the Kolmogorov extension theorem, prediction with and without observation noise, the log marginal likelihood and Occam's razor, hyperparameter fitting, the Cholesky decomposition against naive matrix inversion, and the decision theory used to turn a predictive distribution into a single optimal prediction.

`CovarianceFunction.pdf` develops the theory of covariance functions: Mercer's theorem and eigenfunction analysis, degenerate kernels, the covariance matrix to kernel analogy, Bochner's theorem, eigenvalue decay and smoothness, the Nystrom method and its link to kernel PCA, piecewise polynomial kernels, anisotropy and periodicity, polynomial and neural network kernels, non stationary length scales, the rules for constructing new kernels from old, and a closing survey of open research gaps around non stationary kernels in high dimensional Bayesian optimisation.

## Projects

### 1. Multi Kernel Additive Gaussian Process for Crypto Return Prediction

Directory: `crypto-multikernel-gp`

I built an additive deep kernel Gaussian Process to predict next day cryptocurrency log returns from three separate feature modalities, fundamental, technical, and sentiment, each with its own neural encoder and Matern kernel. The novel contribution is an exact per modality decomposition of the posterior predictive variance, attributing the model's uncertainty at each point to individual modalities and their interactions, with the decomposition holding to machine precision. The model uses a Student t likelihood fitted through variational inference, optional regime gating of modality weights, and a walk forward evaluation protocol with calibration, negative log predictive density, and CRPS against LSTM, GARCH, and single kernel baselines.

### 2. GDP and CO2 Gaussian Process Forecasting

Directory: `gdp-co2-gp-forecasting`

I forecast GDP per capita growth and CO2 per capita to 2030 across ten economies, with the central discipline that every model is scored only on years it never saw during fitting, using log predictive density rather than in sample fit. Each Gaussian Process is required to beat an AR1 and a random walk with drift benchmark before its forecast is treated as meaningful. The project also tests, through an intrinsic coregionalization model, whether GDP and CO2 share usable structure. The main finding is a cautionary one: smooth, confident looking forecasts are frequently not validated once scored honestly, and for almost every country the two series are best modelled separately.

### 3. Laplace Approximation versus Expectation Propagation

Directory: `lpvsep-gp-classification`

Following chapter three of Rasmussen and Williams, I implemented both the Laplace approximation and expectation propagation from scratch for Gaussian Process classification, then compared them head to head on the task of distinguishing handwritten fours from nines, mirroring the digit experiment in section 3.7.3 of the book. This was one of the projects where I learnt the most, because writing the Newton iteration for the Laplace mode and the cavity and moment matching sweeps of expectation propagation by hand forced me to understand every step rather than trust a library. Both methods reach almost identical accuracy, but expectation propagation produces better calibrated probabilities, with lower log loss and Brier score, illustrating why the two schemes diverge most in regions far from the training data.

### 4. MesoNet Deep Kernel Gaussian Process Deepfake Detector

Directory: `mesonet-deepfake-gp`

I paired the compact MesoNet convolutional network with a variational Gaussian Process head using deep kernel learning, so that every classification carries a calibrated variance. Uncertain images are routed to a human review queue rather than forced into a confident answer. This project was a deliberate redesign of an earlier attempt that suffered Gaussian Process collapse, and it addresses each cause of that collapse through constrained kernel hyperparameters, inducing points initialised from real features, and balanced joint optimisation. On the held out test set, roughly eighty four percent of predictions were confident with around eighty five percent accuracy among them, and the remainder were correctly sent to review.

### 5. Multi Fidelity Gaussian Process Surrogate

Directory: `mfgp-multi-fidelity-surrogate`

I explored methods for combining cheap approximate simulations with scarce expensive ones into a single probabilistic model. The notebook progresses from a high fidelity only baseline through the linear autoregressive Kennedy O'Hagan model, the nonlinear autoregressive NARGP model, and finally a deep three tier multi fidelity Gaussian Process, demonstrating along the way why a fixed linear correction breaks down when the correlation between fidelities varies across the input space, and how correctly propagating uncertainty through stacked Gaussian Processes recovers the true function.

### 6. SGEMM Gaussian Process Surrogate

Directory: `sgemm-gp-surrogate`

I built a Gaussian Process surrogate that predicts GPU kernel execution time for single precision matrix multiplication directly from fourteen configuration parameters, avoiding the cost of physically benchmarking every one of the more than two hundred forty thousand possible configurations. Using a Matern kernel with a white noise component the model reaches an R squared of around 0.73, and a second model with an automatic relevance determination kernel ranks which configuration parameters most affect runtime, consistent with known memory and thread tiling bottlenecks.

### 7. Sparse Variational Gaussian Process Transcoding Surrogate

Directory: `svgp-transcoding-surrogate`

I compared an exact Gaussian Process against a sparse variational Gaussian Process for predicting CPU video transcoding time on a dataset of roughly sixty eight thousand rows, where exact inference is infeasible because of its cubic cost. The sparse variational model, trained on the entire dataset with learned inducing points and mini batch optimisation of the evidence lower bound, keeps its uncertainty bands tight and consistent across the full range of configurations, demonstrating the practical advantage of scalable approximate inference over a memory constrained exact baseline.

## Repository Structure

Each project directory contains its own README, notebooks, data handling, and result figures. The `presentations` folder holds the two theory decks. `progress-log.md` records the chronological progression of the internship, and `resources.md` collects the reference material I worked from.

