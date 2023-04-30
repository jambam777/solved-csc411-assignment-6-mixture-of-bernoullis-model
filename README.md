Download Link: https://assignmentchef.com/product/solved-csc411-assignment-6-mixture-of-bernoullis-model
<br>
In this assignment, we’ll implement a probabilistic model which we can apply to the task of image completion. Basically, we observe the top half of an image of a handwritten digit, and we’d like to predict what’s in the bottom half. An example is shown in Figure 1.

This assignment is meant to give you practice with the E-M algorithm. It’s not as long as it looks from this handout. The solution requires about 8-10 lines of code.

<h1>Mixture of Bernoullis model</h1>

The images we’ll work with are all 28 × 28 binary images, <em>i.e. </em>the pixels take values in {0<em>,</em>1}. We ignore the spatial structure, so the images are represented as 784-dimensional binary vectors.

A mixture of Bernoullis model is like the other mixture models we’ve discussed in this course. Each of the mixture components consists of a collection of independent Bernoulli random variables. I.e., conditioned on the latent variable <em>z </em>= <em>k</em>, each pixel <em>x<sub>j </sub></em>is an independent Bernoulli random variable with parameter <em>θ<sub>k,j</sub></em>:

<em>D</em>

<table width="0">

 <tbody>

  <tr>

   <td width="414"><em>p</em>(<strong>x</strong><sup>(<em>i</em>) </sup>|<em>z </em>= <em>k</em>) = <sup>Y </sup><em>p</em>(<em>x<sub>j </sub></em>|<em>z </em>= <em>k</em>)<em>j</em>=1</td>

   <td width="20">(1)</td>

  </tr>

  <tr>

   <td width="414"></td>

   <td width="20">(2)</td>

  </tr>

 </tbody>

</table>

<sup>(<em>i</em>)</sup>

<em>j</em>=1

Try to understand where this formula comes from. You’ll find it useful when you do the derivations.

Given these observations…         … you want to make these predictions

Figure 1: An example of the observed data (left) and the predictions about the missing part of the image (right).

This can be written out as the following generative process:

Sample <em>z </em>from a multinomial distribution with parameter vector <em>π</em>.

For <em>j </em>= 1<em>,…,D</em>:

Sample <em>x<sub>j </sub></em>from a Bernoulli distribution with parameter <em>θ<sub>k,j</sub></em>, where <em>k </em>is the value of <em>z</em>.

It can also be written mathematically as:

<table width="0">

 <tbody>

  <tr>

   <td width="381"><em>z </em>∼ Multinomial(<em>π</em>)</td>

   <td width="20">(3)</td>

  </tr>

  <tr>

   <td width="381"><em>x<sub>j </sub></em>|<em>z </em>= <em>k </em>∼ Bernoulli(<em>θ<sub>k,j</sub></em>)</td>

   <td width="20">(4)</td>

  </tr>

 </tbody>

</table>

<h1>Summary of notation</h1>

We will refer to three dimensions in our model:

<ul>

 <li><em>N </em>= 60,000, the number of training cases. The training cases are indexed by <em>i</em>.</li>

 <li><em>D </em>= 28 × 28 = 784, the dimension of each observation vector. The dimensions are indexed by <em>j</em>.</li>

 <li><em>K</em>, the number of components. The components are indexed by <em>k</em>.</li>

</ul>

The inputs are represented by <strong>X</strong>, an <em>N </em>× <em>D </em>binary matrix. In the E-step, we compute <strong>R</strong>, the matrix of responsibilities, which is an <em>N </em>× <em>K </em>matrix. Each row gives the responsibilities for one training case.

The trainable parameters of the model, written out as vectors and matrices, are:

<em>π</em><sub>1</sub>

<em>π</em>2 <em>π </em>=  … <sub></sub>

<em>π</em><em>K</em>

<table width="0">

 <tbody>

  <tr>

   <td width="84"></td>

   <td width="39"><em>θ</em>1<em>,</em>2 <em>θ</em>2<em>,</em>2</td>

   <td width="33">···…</td>

   <td width="43"><em>θ</em>1<em>,D </em> <em>θ</em>2<em>,D </em>… <sub></sub></td>

  </tr>

 </tbody>

</table>

<em>θ</em><em>K,</em>1         <em>θ</em><em>K,</em>2         ···       <em>θ</em><em>K,D</em>

The rows of <strong>Θ </strong>correspond to mixture components, and columns correspond to input dimensions.

<h1>Part 1: Learning the parameters</h1>

In the first step, we’ll learn the parameters of the model given the responsibilities, using the MAP criterion. This corresponds to the M-step of the E-M algorithm.

In lecture, we discussed the E-M algorithm in the context of maximum likelihood (ML) learning. The MAP case is only slightly different from ML: the only difference is that we add a prior probability term to the objective function in the M-step. In particular, recall that in the context of ML, the M-step maximizes the objective function:

<em>N        K</em>

XX<em>r<sub>k</sub></em>(<em>i</em>) <sup>h</sup>logPr(<em>z</em>(<em>i</em>) = <em>k</em>) + log<em>p</em>(<strong>x</strong>(<em>i</em>) |<em>z</em>(<em>i</em>) = <em>k</em>)<sup>i</sup><em>,                                                      </em>(5)

<em>i</em>=1 <em>k</em>=1

(<em>i</em>) where the <em>r<sub>k </sub></em>are the responsibilities computed during the E-step. In the MAP formulation, we add the (log) prior probability of the parameters:

<em>N        K</em>

logPr(<em>z</em><sup>(<em>i</em>) </sup>= <em>k</em>) + log<em>p</em>(<strong>x</strong><sup>(<em>i</em>) </sup>|<em>z</em><sup>(<em>i</em>) </sup>= <em>k</em>)<sup>i </sup>+ log<em>p</em>(<em>π</em>) + log<em>p</em>(<strong>Θ</strong>)                              (6)

<em>i</em>=1 <em>k</em>=1

Our prior for <strong>Θ </strong>is as follows: every entry is drawn independently from a beta distribution with parameters <em>a </em>and <em>b</em>. The beta distribution is discussed in Lecture 14, but here it is again for reference:

<em>p</em>(<em>θ</em><em>k,j</em>) ∝ <em>θ</em><em>k,ja</em>−1(1 − <em>θ</em><em>k,j</em>)<em>b</em>−1                                                                                                          (7)

Recall that ∝ means “proportional to.” I.e., the distribution has a normalizing constant which we’re ignoring because we don’t need it for the M-step.

For the prior over mixing proportions <em>π</em>, we’ll use the Dirichlet distribution, which is the conjugate prior for the multinomial distribution. It is a distribution over the <strong>probability simplex</strong>, i.e. the set of vectors which define a valid probability distribution.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> The distribution takes the form

<em>.                                                                    </em>(8)

For simplicity, we use a symmetric Dirichlet prior where all the <em>a<sub>k </sub></em>parameters are assumed to be equal. Like the beta distribution, the Dirichlet distribution has a normalizing constant which we don’t need when updating the parameters. The beta distribution is actually the special case of the

Dirichlet distribution for <em>K </em>= 2. You can read more about it on Wikipedia if you’re interested.<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>Your tasks for this part are as follows:

<ol>

 <li><strong> </strong>Derive the M-step update rules for <strong>Θ </strong>and <em>π </em>by setting the partial derivatives of Eqn 6 to zero. Your final answers should have the form:</li>

</ol>

<em>π<sub>k </sub></em>← ··· <em>θ<sub>k,j </sub></em>← ···

Be sure to show your steps. (There’s no trick here; you’ve done very similar questions before.)

<ol start="2">

 <li><strong> </strong>Take these formulas and use them to implement the functions update_pi and Model.update_theta in mixture.py. Each one should be implemented in terms of NumPy matrix and vector operations. Each one requires only a few lines of code, and should not involve any for loops.</li>

</ol>

To help you check your solution, we have provided the function checking.check_m_step. If this check passes, you’re probably in good shape.<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>

To convince us of the correctness of your implementation, <strong>please include the output of running </strong>mixture.print_part_1_values(). Note that we also require you to submit mixture.py through MarkUs.

<ol start="3">

 <li><strong> </strong>The function learn_from_labels learns the parameters of the model from the <em>labeled </em>MNIST images. The values of the latent variables are chosen based on the digit class labels, i.e. the latent variable <em>z</em><sup>(<em>i</em>) </sup>is set to <em>k </em>if the <em>i</em>th training case is an example of digit class <em>k</em>. In terms of the code, this means the matrix <strong>R </strong>of responsibilities has a 1 in the (<em>i,k</em>) entry if the <em>i</em>th image is of class <em>k</em>, and 0 otherwise.</li>

</ol>

Run learn_from_labels to train the model. It will show you the learned components (i.e. rows of <strong>Θ</strong>) and print the training and test log-likelihoods. <em>You do not need to submit anything for this part. It is only for your own satisfaction.</em>

<h1>Part 2: Posterior inference</h1>

Now we derive the posterior probability distribution <em>p</em>(<em>z </em>|<strong>x</strong><sub>obs</sub>), where <strong>x</strong><sub>obs </sub>denotes the subset of the pixels which are observed. In the implementation, we will represent partial observations in terms of variables, where = 1 if the <em>j</em>th pixel of the <em>i</em>th image is observed, and 0 otherwise. In

(<em>i</em>) the implementation, we organize the <em>m<sub>j </sub></em>’s into a matrix <strong>M </strong>which is the same size as <strong>X</strong>.

<ol>

 <li><strong> </strong>Derive the rule for computing the posterior probability distribution <em>p</em>(<em>z </em>|<strong>x</strong>). Your final answer should look something like</li>

</ol>

Pr(<em>z </em>= <em>k </em>|<strong>x</strong>) = ···                                                                           (9)

where the ellipsis represents something you could actually implement. Note that the image may be only partially observed.

<em>Hints: For this derivation, you probably want to express the observation probabilities in the form of Eqn 2.</em>

<ol start="2">

 <li><strong> </strong>Implement the method compute_posterior using your solution to the previous question. While your answer to Question 1 was probably given in terms of probabilities, we do the computations in terms of log probabilities for numerical stability. We’ve already filled in part of the implementation, so your job is to compute log<em>p</em>(<em>z,</em><strong>x</strong>) as described in the method’s doc string.</li>

</ol>

Your implementation should use NumPy matrix and vector operations, rather than a for loop. <em>Hint: There are two lines in </em>Model.log_likelihood <em>which are almost a solution to this question. You can reuse these lines as part of the solution, except you’ll need to modify them to deal with partial observations.</em>

To help you check your solution, we’ve provided the function checking.check_e_step. Note that this check only covers the case where the image is fully observed, so it doesn’t fully verify your solution to this part.

<ol start="3">

 <li>Implement the method posterior_predictive_means, which computes the posterior predictive means of the missing pixels given the observed ones. <em>Hint: this requires only two very short lines of code, one of which is a call to </em>Model.compute_posterior.</li>

</ol>

To convince us of the correctness of the implementation for this part and the previous part, <strong>please include the output of running </strong>mixture.print_part_2_values(). Note that we also require you to submit mixture.py through MarkUs.

<ol start="4">

 <li><strong> </strong>Run the function train_with_em, which trains the mixture model using E-M. It plots the log-likelihood as a function of the number of steps.<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a> You can watch how the mixture components change during training.<a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a> It also shows the model’s image completions after every step. You can watch how they improve over the course of training. At the very end, it outputs the training and test log-likelihoods. The final model for this part should be much better than the one from Part 1. <em>You do not need to submit anything for this part. It’s only for your own satisfaction.</em></li>

</ol>

<h1>Part 3: Conceptual questions</h1>

This section asks you to reflect on the learned model. We tell you the outcomes of the experiments, so that <strong>you can do this part independently of the first 2</strong>. Each question can be answered in a few sentences.

<ol>

 <li><strong> </strong>In the code, the default parameters for the beta prior over <strong>Θ </strong>were <em>a </em>= <em>b </em>= 2. If we instead used <em>a </em>= <em>b </em>= 1 (which corresponds to a uniform distribution), the MAP learning algorithm would have the problem that it might assign zero probability to images in the test set. Why might this happen? <em>Hint: what happens if a pixel is always 0 in the training set, but 1 in the test image?</em></li>

 <li>The model from Part 2 gets significantly higher average log probabilities on both the training and test sets, compared with the model from Part 1. This is counterintuitive, since the Part 1 model has access to additional information: labels which are part of a true causal explanation of the data (i.e. what digit someone was trying to write). Why do you think the Part 2 model still does better?</li>

 <li><strong> </strong>The function print_log_probs_by_digit_class computes the average log-probabilities for different digit classes in both the training and test sets. In both cases, images of 1’s are assigned far higher log-probability than images of 8’s. Does this mean the model thinks 1’s are far more common than 8’s? I.e., if you sample from its distribution, will it generate far more 1’s than 8’s? Why or why not?</li>

</ol>

<a href="#_ftnref1" name="_ftn1">[1]</a> I.e., they must be nonnegative and sum to 1.

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="https://en.wikipedia.org/wiki/Dirichlet_distribution">http://en.wikipedia.org/wiki/Dirichlet_distribution</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> It’s worth taking a minute to think about why this check works. It’s based on the variational interpretation of E-M discussed at the end of Lecture 17. You can also read more about it in Neal and Hinton, 1998, “A view of the

E-M algorithm that justifies incremental, sparse, and other variants.”

<a href="#_ftnref4" name="_ftn4">[4]</a> Observe that it uses a log scale for the number of E-M steps. This is always a good idea, since it can be difficult to tell if the training has leveled off using a linear scale. You wouldn’t know if it’s stopped improving or is just improving very slowly.

<a href="#_ftnref5" name="_ftn5">[5]</a> It’s likely that 5-10 of the mixture components will “die out” during training. In general, this is something we would try to avoid using better initializations and/or priors, but in the context of this assignment it’s the normal behavior.