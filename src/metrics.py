import jax
from jax import numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp
import numpy as np

r"""
    The following summarizes the discussion in Gelman's Bayesian Data Analysis, 3rd ed. Chapter 7.1
    
    Terminology:
    - y: observed data. Used to define posterior. y = {y_1, ..., y_n}.
    - y_{test}: test data. Used to evaluate model performance. y_{test} = {y_{test, 1}, ..., y_{test, n_{test}}}.
    - \tilde{y}: unobserved future data
    - {\theta^{s)}_{s=1}^S: S posterior samples of the model
    - p_{post} = posterior distributon given conditioning on observed data y
    - f = the (unknown) true model, i.e. y \sim f(y)
    
    
    In probabilistic prediction, we aim to report inferences about $\tilde{y}$ in such a way that the full uncertainty
    over $\tilde{y}$ is taken into account. Measures of predictive accuracy for probabilistic prediction are called 
    scoring rules. Common examples of scoring rules include the log-predictive density (i.e.log-likelihood) and the 
    Brier score.
    
    Let us take the log-predictive density as an example.
    
    log pointwise predictive density (lppd) is defined as:
    (7.4) log \prod_{i=1}^n p(y_i | \mathcal{D}) = \sum_{i=1}^n log \int p(y_i | \theta) p(\theta | y) d\theta
    
    as we have access to the posterior samples, we can approximate this value with the computed log pointwise predictive
    density (computed lppd) as:
    (7.5) \sum_{i=1}^n log \Big( \frac{1}{S} \sum_{s=1}^S p(y_i | \theta^{s}) \Big)
    
    Computing (7.5) with the observed data will result in an overestimate of the true lppd. In this work, we focus on 
    the use of a holdout 'test' set y_{test} for a less biased estimate of lppd, by simply replacing y_i with 
    y_{test, i}. In this works we will scale (7.5) by the number of test samples n_{test} for a more interpretable
    per sample lppd:
    (1) \frac{1}{n_{test}} \sum_{i=1}^{n_{test}} log \Big( \frac{1}{S} \sum_{s=1}^S p(y_{test, i} | \theta^{s}) \Big)
    
    Numerical Considerations:
    - Due to the mutual independence assumption - conditioned on model parameters - in this model,  
      p(y_{test, i} | \theta^{s}) = \prod_{j=1}^{E} p(y_{test, i, j} | \theta^{s}). For any reasonably sized |E|, 
      this product can be numerically unstable. To avoid this, we use the log-sum-exp trick - taken over posterior 
      samples - to compute the log-likelihood of the test datum.
      
"""


def bern_log_prob_from_logit(logit: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """

    Numerically stable log probability from logits under a bernoilli likelihood.
        unstable form: y * log sigmoid(logit) + (1 - y) * log (1 - sigmoid(logit))
            - sigmoid can map numbers to ~0 or ~1, which can cause loss of precision.
        stable form: derived in Murphy's Prob Machine Learning Equation (10.13)
            = - log( 1 + exp(-t * logit) ), where t = -1 if y = 0 else +1
            = - softplus(-t * logit)
            = { - softplus(-logit), y == 1
              { - softplus(logit),  y == 0

    Inputs:
        y \in {0, 1}
        logits \in R
        y and logits must be broadcastable. Typically, y will have shape (num_test_samples, num_edges) and logits will
        have shape (num_posterior_samples, num_test_samples, num_edges).

    Returns:
        log prob of logits, jnp.ndarray sample shape as logits

    """
    return jnp.where(y, -jax.nn.softplus(-logit), -jax.nn.softplus(logit))


def log_predictive_density(edge_logits: jnp.ndarray, y: jnp.ndarray):
    """
        Compute the log predictive density of *each* of the num_data_samples datum given the model.
        Assuming use of Bernoulli predictive likelihood.

        Consider the log predictive density for single datum y_i:
        log \int p(y_i | \theta) p(\theta | y) d\theta
        = log \int \prod_{j=1}^{E} p(y_{i, j} | \theta) p(\theta | y) d\theta <-- mutual independence assumption on edges given parameters
        \approx log \Big( \frac{1}{S} \sum_{s=1}^S \prod_{j=1}^{E} p(y_{i, j} | \theta^{s}) \Big)
        = -log(S) + log \sum_{s=1}^S \prod_{j=1}^{E} p(y_{i, j} | \theta^{s})
        = -log(S) + logsumexp(\sum_{j=1}^{E} log p(y_{i, j} | \theta^{s}))_{s=1}^S <-- log-sum-exp trick

        Inputs:
            edge_logits: shape (num_posterior_samples, num_data_samples, num_edges)
                - sigmoid(edge_probs[s, i, j]) is the probability that the j-th edge is present in the i-th datum
                  under the s-th posterior sample.
            y: shape (num_data_samples, num_edges)
                - y[i, j] is the observed value of the j-th edge in the i-th data sample.
    """

    num_posterior_samples, num_data_samples, num_edges = edge_logits.shape
    post_samples_ax, data_samples_ax, edges_ax = 0, 1, 2

    # step 1: compute the log prob of each edge under each posterior sample: log p(y_{i, j} | \theta^{s})
    edge_log_probs = bern_log_prob_from_logit(edge_logits, y)

    # step 2: sum over edges j: \sum_{j=1}^{E} log p(y_{i, j} | \theta^{s}) term
    sum_edge_log_probs = jnp.sum(edge_log_probs, axis=edges_ax)

    # step 3: log-sum-exp over posterior samples s
    # = logsumexp(\sum_{j=1}^{E} log p(y_{i, j} | \theta^{s}))_{s=1}^S - log(S)
    log_pred_density = logsumexp(sum_edge_log_probs, axis=post_samples_ax) - jnp.log(num_posterior_samples)

    return log_pred_density


def brier_score(edge_logits: jnp.ndarray, y: jnp.ndarray):
    """
        Compute the Brier Score of *each* of the num_data_samples datum.
        Brier score of sample y_i | \theta^2 = \frac{1}{E}  \| p(y_i | \theta^{s}) - y_i\|_2^2

        The Brier Score for datum y_i \in {0, 1}^E is defined as:
        B(y_i | y)
        = \int \frac{1}{E} \| p(y_i | \theta^{s}) - y_i \||_2^2 p(\theta | y) d\theta
        \approx \frac{1}{S} \sum_{s=1}^S \frac{1}{E} \| p(y_i | \theta^{s}) - y_i \||_2^2  <-- Monte Carlo approximation
        = \frac{1}{E*S} \sum_{s=1}^S  \| p(y_i | \theta^{s}) - y_i \||_2^2  <-- squaring small number can cause problems
        = exp [ log (  \sum_{s=1}^S  \| p(y_i | \theta^{s}) - y_i \||_2^2 ) ) - log(S) - log(E) ]
        = exp [ logsumexp{ 2*log( |p(y_{i,j} | \theta^{s}) - y_{i,j) | }_{s, j}  - log(S) - log(E) ]


        Inputs:
            edge_logits: shape (num_posterior_samples, num_data_samples, num_edges)
                - sigmoid(edge_probs[s, i, j]) is the probability that the j-th edge is present in the i-th datum
                  under the s-th posterior sample.
            y: shape (num_data_samples, num_edges)
                - y[i, j] is the observed value of the j-th edge in the i-th data sample.
    """
    num_posterior_samples, num_data_samples, num_edges = edge_logits.shape
    post_samples_ax, data_samples_ax, edges_ax = 0, 1, 2
    # direct_brier_score = jnp.mean((jax.nn.sigmoid(logit) - y) ** 2, axis=(post_samples_ax, edges_ax))

    edge_probs = jax.nn.sigmoid(edge_logits)
    log_brier_score = logsumexp(2*jnp.log(jnp.abs(edge_probs - y)), axis=(post_samples_ax, edges_ax)) \
                      - jnp.log(num_posterior_samples) - jnp.log(num_edges)
    return jnp.exp(log_brier_score)


def calibration(edge_probs: jnp.ndarray, y: jnp.ndarray, num_bins: int):
    """
        Compute the calibration of the model.

        Following the methodoloy laid out in "On the Calibration of Modern Neural Networks" by Guo et al.

        Note that in binary classification using a threshold of 0.5, confidence will never be < 0.5. Thus the
        initial bin edge should be 0.5.

        edge_probs: mean of posterior predictive, shape (num_test_samples, num_edges)
        y: labels, shape (num_test_samples, num_edges)
        bin_edges: edges of the bins to use for calibration curve
    """
    assert num_bins > 1
    assert edge_probs.ndim == 2 and y.ndim == 2, "edge_probs and y must be 2D arrays. Edge_probs is mean of " \
                                                 "predictive posterior."
    y = np.reshape(y, (-1,))
    edge_probs = np.reshape(edge_probs, (-1,))

    # for numerical reasons, slightly expand the first and last bin to include edge_probs that are exactly .5 and 1
    bin_edges = np.linspace(0.5, 1, num_bins)
    bin_edges[0], bin_edges[-1] = bin_edges[0] - 1e-1, bin_edges[-1] + 1e-1

    # choose threshold of 0.5 to predict positive
    discrete_preds = np.where(edge_probs > 0.5, 1, 0)

    # confidence of a prediction is the probability of the *predicted* class: \hat{p}_n
    # if we predict 1, then confidence is p_n, if we predict 0, then confidence is 1 - p_n
    edge_confidence = np.where(discrete_preds, edge_probs, 1 - edge_probs)

    # Construct Bins B_m: assign each edge to a bin based on its confidence:
    bin_idxs = np.digitize(edge_confidence, bin_edges)# - 1
    # print the proportion of edges in each bin
    #print(np.bincount(bin_idxs, minlength=num_bins) / len(bin_idxs))

    # accuracy within bin b is the fraction of edges in bin b that are correctly predicted. We will first count
    # how many total correct predictions we make, then divide by the total number of edges in each bin to find the
    # accuracy within each bin.
    # acc(\mathcal{B}_b) in Adv. PML

    # the average confidence within a bin is the average edge confidence of edges in bin b.
    # conf(\mathcal{B}_b) = \mean_{n \in \mathcal{B}_b} \hat{p}_n

    bin_counts, bin_total_prob, bin_total_positive, bin_total_correct, bin_total_confidence \
        = np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins)
    for i in range(num_bins):
        bin_mask = bin_idxs == i
        bin_counts[i] = np.sum(bin_mask)
        bin_total_positive[i] = np.sum(y[bin_mask])
        bin_total_correct[i] = np.sum(y[bin_mask] == discrete_preds[bin_mask])
        bin_total_prob[i] = np.sum(edge_probs[bin_mask])
        bin_total_confidence[i] = np.sum(edge_confidence[bin_mask]) if np.sum(bin_mask) else 0.0

    # if bin_counts == 0 -> produces nan. Set to 0, b/c bin_frac will kill any value that occurs anyway
    mask = bin_counts > 0
    bin_acc, bin_ave_confidence, bin_frac_pos, bin_ave_prob \
        = np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins)
    bin_acc[mask] = bin_total_correct[mask] / bin_counts[mask]
    bin_ave_confidence[mask] = bin_total_confidence[mask] / bin_counts[mask]
    bin_frac_pos[mask] = bin_total_positive[mask] / bin_counts[mask]
    bin_ave_prob[mask] = bin_total_prob[mask] / bin_counts[mask]

    assert len(y) == np.sum(bin_counts), f'not all edges assigned to a bin: len(y) {len(y)} != sum of bin counts ' \
                                         f'{np.sum(bin_counts)}'

    bin_frac = bin_counts / len(y)
    ece = np.sum(np.abs(bin_acc - bin_ave_confidence) * bin_frac)
    ece_raw = (1/len(y)) * np.sum(np.abs(bin_total_correct - bin_total_confidence))

    # reliability diagram is fraction of positive edges in bin (y) vs. average probability of edges in bin (x)

    return {'ece': ece,
            'bin_counts': bin_counts,
            'bin_acc': bin_acc,
            'bin_confidence': bin_ave_confidence, # prob_pred output in sklearn.calibration.calibration_curve
            'ave_prob': bin_ave_prob,
            'frac_pos': bin_frac_pos} # prob_true output in sklearn.calibration.calibration_curve


def compute_metrics(edge_logits: jnp.ndarray, y: jnp.ndarray, num_bins: int):
    """
        edge_logits: (num_posterior_samples, num_test_samples, num_edges)
        y: (num_test_samples, num_edges)
        num_bins: number of uniform bins to use for calibration curve
    """
    assert edge_logits.ndim == 3 and y.ndim == 2
    assert edge_logits.shape[1:] == y.shape
    num_posterior_samples, num_test_samples, num_edges = edge_logits.shape
    post_samples_ax, test_samples_ax, edges_ax = 0, 1, 2

    edge_probs = jax.nn.sigmoid(edge_logits)

    # use the mean of the posterior predictive as the prediction
    edge_probs_post_pred_mean = jnp.mean(edge_probs, axis=post_samples_ax)

    # accuracy
    edge_correct_predictions = (jnp.where(edge_probs_post_pred_mean > 0.5, 1, 0) == y) + 0.0
    accuracies = jnp.mean(edge_correct_predictions, axis=-1)

    # calibration
    calibration_dict = calibration(edge_probs_post_pred_mean, y, num_bins)

    return {'log_likelihoods': log_predictive_density(edge_logits, y),
            'brier_scores': brier_score(edge_logits, y),
            'accuracies': accuracies,
            'calibration_dict': calibration_dict}


def bernoulli_variance_from_logit(logits: jnp.ndarray) -> jnp.ndarray:
    """
    Variance of the bernoulli each logit defines, i.e. \sigmoid(logits)(1 - \sigmoid(logits)), but in numerically stable way

    Inputs:
        y \in {0, 1}
        logits \in R

    Returns:
        log prob of logits, jnp.ndarray sample shape as logits
    """
    log_term = -2 * jnp.logaddexp(0, -logits) - logits
    variance = jnp.exp(log_term)
    return variance


if __name__ == '__main__':
    """
        Tests for numerically stable log prob via softplus
    """
    def bernoulli_log_likelihood(p, y):
        return y * jnp.log(p) + (1 - y) * jnp.log(1 - p)

    # for logits between ~[-2, 2] outputs should be identical
    y = jnp.array(    [0,  1, 0, 1,  0,  1,   0,  1, 0, 1])
    logit = jnp.array([-2, -2, 2, 2, -1, 1, -.5, .5, 0, 1])
    assert jnp.allclose(bernoulli_log_likelihood(jax.nn.sigmoid(logit), y),
                        bern_log_prob_from_logit(logit, y))

    # when logits become extreme, the softplus version should be more numerically stable
    y = jnp.array([0.0, 0, 1, 1])
    logit = jnp.array([-1.0, 1, -1, 1])
    for scale in [10, 20, 100]:
        scaled_logit = scale * logit
        direct = bernoulli_log_likelihood(jax.nn.sigmoid(scaled_logit), y)
        softplus = bern_log_prob_from_logit(scaled_logit, y)
        abs_diff = jnp.abs(direct - softplus)
        if jnp.any(jnp.isnan(direct)) or jnp.any(jnp.isinf(direct)):
            print(f'scale {scale}: nan or inf detected')
            nan_mask = jnp.isnan(direct) | jnp.isnan(softplus)
            #print(f'\tnan labels y {y[nan_mask]}')
            #print(f'\tnan logits {scaled_logit[nan_mask]}')
            print(f'\tdirect numerically unstable {direct[nan_mask]}')
            print(f'\tsoftplus numerically stable {softplus[nan_mask]}')

    # ensure y broadcasts properly for bern_log_prob_from_logit
    # y.shape = [num_test_samples, num_edges]
    # logit.shape = [num_posterior_samples, num_test_samples, num_edges]
    # output.shape = [num_posterior_samples, num_test_samples, num_edges]
    num_posterior_samples, num_test_samples, num_edges = 4000, 100, 190
    y = jax.random.randint(jax.random.PRNGKey(0), shape=(num_test_samples, num_edges), minval=0, maxval=2)
    logit = jax.random.normal(jax.random.PRNGKey(1), shape=(num_test_samples, num_edges))
    logit = jnp.tile(logit, (num_posterior_samples, 1, 1))
    output = bern_log_prob_from_logit(logit, y)
    # check each output[i] is the same
    assert jnp.allclose(output[0], output), 'since logits are the same for each posterior sample (by construction), ' \
                                            'the output should be the same for each posterior sample'

    """
        Tests for log predictive density
    """
    all_ones_y = jnp.ones((num_test_samples, num_edges))
    all_zeros_y = jnp.zeros((num_test_samples, num_edges))

    # test case 1: all edges are 1 (0) and we predict prob 1 (0) for each edge
    assert jnp.allclose(0.0,
                        log_predictive_density(
                            1e20*jnp.ones((num_posterior_samples, num_test_samples, num_edges)),  # logits
                            all_ones_y))
    assert jnp.allclose(0.0,
                        log_predictive_density(
                            -1e20 * jnp.ones((num_posterior_samples, num_test_samples, num_edges)),  # logits
                            all_zeros_y))

    # test case 2: all edges 1 (0), and we predict prob = .5 for all edges
    assert jnp.allclose(
        # truth: log p(y | \theta) = E * (y log p + (1-y) log(1-p)) = num_edges * 1 * log(.5) b/c y = 1, l = 0
        # over entire posterior: logsumexp(log p(y | \theta^s))_s - log(S)
        logsumexp(num_edges * jnp.log(.5) * jnp.ones(num_posterior_samples)) - jnp.log(num_posterior_samples),
        log_predictive_density(jnp.zeros((num_posterior_samples, num_test_samples, num_edges)), all_ones_y))
    assert jnp.allclose(
        # truth: log p(y | \theta) = E * (y log p + (1-y) log(1-p)) = num_edges * 1 * log(.5) b/c y = 1, l = 0
        # over entire posterior: logsumexp(log p(y | \theta^s))_s - log(S)
        logsumexp(num_edges * jnp.log(.5) * jnp.ones(num_posterior_samples)) - jnp.log(num_posterior_samples),
        log_predictive_density(jnp.zeros((num_posterior_samples, num_test_samples, num_edges)), all_zeros_y))

    """
        Tests for Brier score
    """
    num_posterior_samples, num_test_samples, num_edges = 4000, 100, 190
    y = jax.random.randint(jax.random.PRNGKey(0), shape=(num_test_samples, num_edges), minval=0, maxval=2)
    # test 1: well-behaved logits -> probabilities should be close to 0.5 -> (y - p)^2 should be well behaved
    logit = jax.random.normal(jax.random.PRNGKey(1), shape=(num_test_samples, num_edges))
    logit = jnp.tile(logit, (num_posterior_samples, 1, 1))
    direct = jnp.mean((jax.nn.sigmoid(logit) - y) ** 2, axis=(0, -1)) # ave squared error over posterior samples and edges
    stable = brier_score(logit, y)
    assert jnp.allclose(direct, stable), 'stable brier score should be the same as the direct computation for this case'

    # test case 2: extreme logits -> probabilities should be close to 0 or 1 -> (y - p)^2 msy underflow
    logit_ = jax.random.randint(jax.random.PRNGKey(0), shape=(num_posterior_samples, num_test_samples, num_edges),
                                minval=1, maxval=3) / 3
    """
    for s in [1, 5, 10, 20, 30, 40, 50]:
        logit = jnp.exp(s) * logit_ #jnp.ones((num_posterior_samples, num_test_samples, num_edges))
        direct = jnp.mean((jax.nn.sigmoid(logit) - y) ** 2, axis=(0, -1))
        stable = brier_score(logit, y)
        diff = jnp.abs(direct - stable).max()
        print(f'scale {s}: max diff between direct and stable brier score for extreme logits: {diff}')
    """

    """
        Tests for Variance of Bernoulli from Logits
    """
    # Test cases
    # Test for very large logit
    large_logit = jnp.array([1e6])
    assert jnp.isclose(bernoulli_variance_from_logit(large_logit), 0, atol=1e-5)

    # Test for very small logit
    small_logit = jnp.array([-1e6])
    assert jnp.isclose(bernoulli_variance_from_logit(small_logit), 0, atol=1e-5)

    # Test for logit = 0
    zero_logit = jnp.array([0])
    assert jnp.isclose(bernoulli_variance_from_logit(zero_logit), 0.25, atol=1e-5)

    # Test for logit = ±1.09861
    positive_logit = jnp.array([1.09861])
    negative_logit = jnp.array([-1.09861])
    expected_variance = 0.1875
    assert jnp.isclose(bernoulli_variance_from_logit(positive_logit), expected_variance, atol=1e-5)
    assert jnp.isclose(bernoulli_variance_from_logit(negative_logit), expected_variance, atol=1e-5)

