# ============================================================
#  Choice probabilities in Rational Inattention models
#  with Shannon (mutual information) cost functions
#  Daniel Csaba
#  February, 2017
# ============================================================



# ============================================================
#       Solving for the optimal information structure
#
# Using the Blahut-Arimoto algorithm to solve for the optimal
# information structure.
# See Rate Distortion Function in Cover & Thomas (2006).
# ============================================================


import sys
import numpy as np

class Blahut_Arimoto:
    """ Solve for optimal information structure
    in Rational Inattention models with Shannon
    cost function."""

    def __init__(self, U, k, mu):
        """
        :param U: utility function given by payoff matrix (statesXactions)
        :param k: multiplier on Shannon cost function
        :param p: prior probability over states

        :return: Return object with optimal information structure for information processing.
        """

        self.U, self.k, self.mu = U, k, mu
        # Number of states
        # self.num_state = U.shape[0]
        # Number of actions
        # self.num_action = U.shape[1]
        # Check compatibility of U and p
        if U.shape[0] != len(mu):
            sys.exit("The dimensions of the prior probability vector, p, are not aligned with payoff matrix.")

        self.opt_exp = self._opt_prob()

    def _opt_prob(self):
        """Computes the unconditional choice probabilities."""

        # Transformed utilities
        U_trans = (np.e ** (self.U / self.k))

        # Initial guess for unconditional choice probabilities
        p = np.random.uniform(0, 1, self.U.shape[1])

        tol = 1.0e-15    # set tolerance level
        dist = 100       # set initial distance

        while dist > tol:

            # Compute the optimal experiment
            Exper = ((U_trans * p).T / ((U_trans * p).sum(1))).T
            # Compute the updated unconditional probabilities
            p_new = self.mu @ Exper
            # Compute the distance
            dist = np.linalg.norm(p - p_new)
            p = p_new # update

        return ((U_trans * p).T / ((U_trans * p).sum(1))).T




# -------------------------------
#    Class attributes starting
# -------------------------------

    np.set_printoptions(precision=8, suppress=True)


    @property
    def unconditional_prob(self):
        """Returns the unconditional choice probabilities."""
        return self.mu @ self.opt_exp

    @property
    def conditional_prob(self):
        """Returns the conditional choice probabilities
        i.e. the optimal experiment."""
        return self.opt_exp

    @property
    def opt_posterior(self):
        """Returns the optimal posteriors corresponding
        to each action."""
        return (self.opt_exp.T * self.mu).T / (self.mu @ self.opt_exp)
