# -*- coding: utf-8 -*-
import numpy as np

def wolfe(feval, xk, dk, gk, ck, alpha):
    rho = 1e-4
    sigma = 0.9  # sigma control the weight between exact and inexact line seach,
    max_iter = 100
    a = 0.
    b = float('inf')
    g_old = gk
    v_old = ck
    gtd_old = np.dot(g_old.T, dk)
    for i in xrange(max_iter):
        xk_new = xk + alpha * dk
        v_new, g_new = feval(xk_new)
        gtd_new = np.dot(g_new.T, dk)
        if not v_new <= v_old + rho * alpha * gtd_old:
            b = alpha
            alpha = (alpha + a) / 2
            continue
        if not abs(gtd_new) <= -sigma * gtd_old:
            if gtd_new * (b - a) >= 0:
                b = a
            a = alpha
            alpha = min(10 * alpha, (b + alpha) / 2)
            continue
        break
    return alpha, v_new, g_new, xk_new
