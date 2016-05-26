# -*- coding: utf-8 -*-
import numpy as np


def wolfe(f, xk, dk, alpha, rho=0.1, sigma=0.4, max_iter=50):
    # rho control the distance from right interval, rho ∈ (0, 0.5)
    # sigma control the weight between exact and inexact line seach, sigma ∈ [rho, 1]
    # sigma = 0.1 indicates a very exact line seach which would time consuming
    # sigma = 0.9 indicates a weak line search(inexact), but more faster
    a = 0.
    b = float('inf')
    ck, gk = f(xk)
    gtd = np.dot(gk.T, dk)
    for i in xrange(max_iter):
        xk_plus = xk + alpha * dk
        ck_plus, gk_plus = f(xk_plus)

        # condition: f(xk + alpha*dk) <= f(xk) + rho*f'(xk)*alpha*dk
        if not ck_plus <= ck + rho * alpha * gtd:
            _alpha = a + (alpha - a) / (2 * (1 + (ck - ck_plus) / ((alpha - a) * gtd)))
            b = alpha
            alpha = _alpha
            continue

        gtd_plus = np.dot(gk_plus.T, dk)
        # condition: |f'(xk+alpha*dk)*dk| <= - sigma * f'(xk) * dk
        if not abs(gtd_plus) <= -sigma * gtd:
            if gtd_plus * (b - a) >= 0:
                b = a
            a = alpha
            alpha = min(10 * alpha, (b + alpha) / 2)
            continue
        break
    return alpha, ck_plus, gk_plus, xk_plus


def armijo(f, xk, dk, alpha, rho=1e-1, max_iter=50):
    # rho control the distance from right interval, rho ∈ (0, 0.5)
    a = 0.
    b = float('inf')
    ck, gk = f(xk)
    gtd = np.dot(gk.T, dk)
    for i in xrange(max_iter):
        xk_plus = xk + alpha * dk
        ck_plus, gk_plus = f(xk_plus)

        # condition: f(xk + alpha*dk) <= f(xk) + rho*f'(xk)*alpha*dk
        if not ck_plus <= ck + rho * alpha * gtd:
            b = alpha
            alpha = (a + b) / 2
            continue

        # condition: f(xk + alpha*dk) >= f(xk) + (1-rho)*f'(xk)*alpha*dk
        if not ck_plus >= ck + (1 - rho) * alpha * gtd:
            a = alpha
            alpha = min(10 * alpha, (a + b) / 2)
            continue
        break
    return alpha, ck_plus, gk_plus, xk_plus
