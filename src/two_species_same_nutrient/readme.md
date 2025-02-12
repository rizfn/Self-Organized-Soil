# Mean Field equations

$$
\begin{align}
\frac{\mathrm{d}S}{\mathrm{d}t} &= \sigma S (E + N) - S(W_G + W_B) \\
\frac{\mathrm{d}E}{\mathrm{d}t} &= S \big[ (1-\mu_1) W_G + (1-\mu_2) W_B \big]  + N \big[(1-\rho_1) W_G + (1-\rho_2) W_B \big]  + \theta \left(W_G+W_B\right) - \sigma S E \\
\frac{\mathrm{d}N}{\mathrm{d}t} &= S(\mu_1 W_G + \mu_2 W_G) - N(W_G + W_B) - \sigma S N \\
\frac{\mathrm{d}W_G}{\mathrm{d}t} &= \rho_1 W_G N - \theta W_G \\
\frac{\mathrm{d}W_B}{\mathrm{d}t} &= \rho_2 W_B N - \theta W_B
\end{align}
$$



Fixed points:

$$
\begin{align*}
    S &= 0, & \text{Em} &= 1-N, & G &= 0, & B &= 0 \\
    S &= \frac{-\sqrt{(\theta  \sigma -\text{$\mu $1} \text{$\rho $1})^2-4 \theta  \text{$\mu $1} \text{$\rho $1}}-\theta  \sigma +\text{$\mu $1} \text{$\rho $1}}{2 \text{$\mu $1} \text{$\rho $1}}, & \text{Em} &= \frac{\frac{\sqrt{(\theta  \sigma -\text{$\mu $1} \text{$\rho $1})^2-4 \theta  \text{$\mu $1} \text{$\rho $1}}}{\text{$\mu $1}}+\frac{\theta  \sigma }{\text{$\mu $1}}-2 \theta  \sigma -2 \theta +\text{$\rho $1}}{2 (\text{$\rho $1} \sigma +\text{$\rho $1})}, & N &= \frac{\theta }{\text{$\rho $1}}, & G &= \frac{\frac{\theta  \sigma ^2}{\text{$\mu $1} \text{$\rho $1}}+\frac{\sigma  \sqrt{(\theta  \sigma -\text{$\mu $1} \text{$\rho $1})^2-4 \theta  \text{$\mu $1} \text{$\rho $1}}}{\text{$\mu $1} \text{$\rho $1}}+\sigma }{2 (\sigma +1)}, & B &= 0 \\
    S &= \frac{\sqrt{(\theta  \sigma -\text{$\mu $1} \text{$\rho $1})^2-4 \theta  \text{$\mu $1} \text{$\rho $1}}-\theta  \sigma +\text{$\mu $1} \text{$\rho $1}}{2 \text{$\mu $1} \text{$\rho $1}}, & \text{Em} &= \frac{-\frac{\sqrt{(\theta  \sigma -\text{$\mu $1} \text{$\rho $1})^2-4 \theta  \text{$\mu $1} \text{$\rho $1}}}{\text{$\mu $1}}+\frac{\theta  \sigma }{\text{$\mu $1}}-2 \theta  \sigma -2 \theta +\text{$\rho $1}}{2 (\text{$\rho $1} \sigma +\text{$\rho $1})}, & N &= \frac{\theta }{\text{$\rho $1}}, & G &= \frac{\frac{\theta  \sigma ^2}{\text{$\mu $1} \text{$\rho $1}}-\frac{\sigma  \sqrt{(\theta  \sigma -\text{$\mu $1} \text{$\rho $1})^2-4 \theta  \text{$\mu $1} \text{$\rho $1}}}{\text{$\mu $1} \text{$\rho $1}}+\sigma }{2 (\sigma +1)}, & B &= 0 \\
    S &= \frac{-\sqrt{(\theta  \sigma -\text{$\mu $2} \text{$\rho $2})^2-4 \theta  \text{$\mu $2} \text{$\rho $2}}-\theta  \sigma +\text{$\mu $2} \text{$\rho $2}}{2 \text{$\mu $2} \text{$\rho $2}}, & \text{Em} &= \frac{\frac{\sqrt{(\theta  \sigma -\text{$\mu $2} \text{$\rho $2})^2-4 \theta  \text{$\mu $2} \text{$\rho $2}}}{\text{$\mu $2}}+\frac{\theta  \sigma }{\text{$\mu $2}}-2 \theta  \sigma -2 \theta +\text{$\rho $2}}{2 (\text{$\rho $2} \sigma +\text{$\rho $2})}, & N &= \frac{\theta }{\text{$\rho $2}}, & G &= 0, & B &= \frac{\frac{\theta  \sigma ^2}{\text{$\mu $2} \text{$\rho $2}}+\frac{\sigma  \sqrt{(\theta  \sigma -\text{$\mu $2} \text{$\rho $2})^2-4 \theta  \text{$\mu $2} \text{$\rho $2}}}{\text{$\mu $2} \text{$\rho $2}}+\sigma }{2 (\sigma +1)} \\
    S &= \frac{\sqrt{(\theta  \sigma -\text{$\mu $2} \text{$\rho $2})^2-4 \theta  \text{$\mu $2} \text{$\rho $2}}-\theta  \sigma +\text{$\mu $2} \text{$\rho $2}}{2 \text{$\mu $2} \text{$\rho $2}}, & \text{Em} &= \frac{-\frac{\sqrt{(\theta  \sigma -\text{$\mu $2} \text{$\rho $2})^2-4 \theta  \text{$\mu $2} \text{$\rho $2}}}{\text{$\mu $2}}+\frac{\theta  \sigma }{\text{$\mu $2}}-2 \theta  \sigma -2 \theta +\text{$\rho $2}}{2 (\text{$\rho $2} \sigma +\text{$\rho $2})}, & N &= \frac{\theta }{\text{$\rho $2}}, & G &= 0, & B &= \frac{\frac{\theta  \sigma ^2}{\text{$\mu $2} \text{$\rho $2}}-\frac{\sigma  \sqrt{(\theta  \sigma -\text{$\mu $2} \text{$\rho $2})^2-4 \theta  \text{$\mu $2} \text{$\rho $2}}}{\text{$\mu $2} \text{$\rho $2}}+\sigma }{2 (\sigma +1)} \\
    S &= 0, & \text{Em} &= \frac{\text{$\rho $1}-\theta }{\text{$\rho $1}}, & N &= \frac{\theta }{\text{$\rho $1}}, & G &= 0, & B &= 0 \\
    S &= 0, & \text{Em} &= \frac{\text{$\rho $2}-\theta }{\text{$\rho $2}}, & N &= \frac{\theta }{\text{$\rho $2}}, & G &= 0, & B &= 0 \\
    S &= 1, & \text{Em} &= 0, & N &= 0, & G &= 0, & B &= 0
\end{align*}
$$


## Longer Living Parasite (aka `different_thetas`):

$$
\begin{align}
\frac{\mathrm{d}S}{\mathrm{d}t} &= \sigma S (E + N) - S(W_G + W_B) \\
\frac{\mathrm{d}E}{\mathrm{d}t} &= S W_B + \theta_1 W_G + \theta_2W_B - \sigma S E \\
\frac{\mathrm{d}N}{\mathrm{d}t} &= S W_G - N(W_G + W_B) - \sigma S N \\
\frac{\mathrm{d}W_G}{\mathrm{d}t} &= W_G N - \theta_1 W_G \\
\frac{\mathrm{d}W_B}{\mathrm{d}t} &= W_B N - \theta_2 W_B
\end{align}
$$

Fixed points:

$$
\begin{align*}
    S &= 0, & \text{Em} &= 1-N, & G &= 0, & B &= 0 \\
    S &= 0, & \text{Em} &= 1-\theta_1, & N &= \theta_1, & G &= 0, & B &= 0 \\
    S &= 0, & \text{Em} &= 1-\theta_2, & N &= \theta_2, & G &= 0, & B &= 0 \\
    S &= 1, & \text{Em} &= 0, & N &= 0, & G &= 0, & B &= 0 \\
    S &= -\frac{1}{\sigma}, & \text{Em} &= \frac{1-\theta_2 \sigma}{\sigma}, & N &= \theta_2, & G &= 0, & B &= 1 \\
    S &= \frac{1}{2} \left(-\theta_1 \sigma -\sqrt{(\theta_1 \sigma -1)^2-4 \theta_1}+1\right), & \text{Em} &= \frac{\theta_1 (-\sigma) + \sqrt{(\theta_1 \sigma -1)^2-4 \theta_1} - 2 \theta_1 + 1}{2 (\sigma + 1)}, & N &= \theta_1, & G &= \frac{\theta_1 \sigma^2 + \sigma \sqrt{(\theta_1 \sigma -1)^2-4 \theta_1} + \sigma}{2 (\sigma + 1)}, & B &= 0 \\
    S &= \frac{1}{2} \left(-\theta_1 \sigma + \sqrt{(\theta_1 \sigma -1)^2-4 \theta_1} + 1\right), & \text{Em} &= \frac{\theta_1 (-\sigma) - \sqrt{(\theta_1 \sigma -1)^2-4 \theta_1} - 2 \theta_1 + 1}{2 (\sigma + 1)}, & N &= \theta_1, & G &= \frac{\theta_1 \sigma^2 - \sigma \sqrt{(\theta_1 \sigma -1)^2-4 \theta_1} + \sigma}{2 (\sigma + 1)}, & B &= 0
\end{align*}
$$


## Longer Living Parasite, which **doesn't** destroy soil:

$$
\begin{align}
\frac{\mathrm{d}S}{\mathrm{d}t} &= \sigma S (E + N) - SW_G \\
\frac{\mathrm{d}E}{\mathrm{d}t} &= \theta_1 W_G + \theta_2W_B - \sigma S E \\
\frac{\mathrm{d}N}{\mathrm{d}t} &= S W_G - N(W_G + W_B) - \sigma S N \\
\frac{\mathrm{d}W_G}{\mathrm{d}t} &= W_G N - \theta_1 W_G \\
\frac{\mathrm{d}W_B}{\mathrm{d}t} &= W_B N - \theta_2 W_B
\end{align}
$$

Fixed points:

Fixed points:

$$
\begin{align*}
    S &= 0, & \text{Em} &= 1 - N, & G &= 0, & B &= 0 \\
    S &= 1, & \text{Em} &= 0, & N &= 0, & G &= 0, & B &= 0 \\
    S &= \frac{1}{2} \left(1 - \theta_1 \sigma - \sqrt{-4 \theta_1 + (-1 + \theta_1 \sigma)^2}\right), & \text{Em} &= \frac{1 - 2 \theta_1 - \theta_1 \sigma + \sqrt{-4 \theta_1 + (-1 + \theta_1 \sigma)^2}}{2 (1 + \sigma)}, & N &= \theta_1, & G &= \frac{\sigma + \theta_1 \sigma^2 + \sigma \sqrt{-4 \theta_1 + (-1 + \theta_1 \sigma)^2}}{2 (1 + \sigma)}, & B &= 0 \\
    S &= \frac{1}{2} \left(1 - \theta_1 \sigma + \sqrt{-4 \theta_1 + (-1 + \theta_1 \sigma)^2}\right), & \text{Em} &= \frac{1 - 2 \theta_1 - \theta_1 \sigma - \sqrt{-4 \theta_1 + (-1 + \theta_1 \sigma)^2}}{2 (1 + \sigma)}, & N &= \theta_1, & G &= \frac{\sigma + \theta_1 \sigma^2 - \sigma \sqrt{-4 \theta_1 + (-1 + \theta_1 \sigma)^2}}{2 (1 + \sigma)}, & B &= 0 \\
    S &= \frac{1}{1 - \sigma}, & \text{Em} &= -\theta_2, & N &= \theta_2, & G &= 0, & B &= \frac{\sigma}{-1 + \sigma} \\
    S &= 0, & \text{Em} &= 1 - \theta_1, & N &= \theta_1, & G &= 0, & B &= 0 \\
    S &= 0, & \text{Em} &= 1 - \theta_2, & N &= \theta_2, & G &= 0, & B &= 0
\end{align*}
$$
