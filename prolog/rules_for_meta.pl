dividedBy(2,1) if true.
dividedBy(6,3) if true.
dividedBy(6,2) if true.
dividedBy(4,2) if true.
dividedBy(8,4) if true.
dividedBy(B, A) if dividedBy(B, C), dividedBy(C, A).
dividedBy(C, A) if inv_divides(B, A), dividedBy(C, B).
% 0.165 (0.345)
dividedBy(C, A) if inv_divides(B, A), inv_divides(C, B).
% 0.137 (0.287)
dividedBy(B, A) if dividedBy(B, A).
% 0.088 (0.185)
dividedBy(B, A) if inv_divides(B, A).
% 0.047 (0.099)
dividedBy(B, A) if inv_divides(B, A).
% 0.046 (0.097)
dividedBy(C, A) if dividedBy(B, A), dividedBy(C, B).
% 0.016 (0.034)
dividedBy(C, A) if dividedBy(B, A), inv_divides(C, B).
% 0.014 (0.029)
dividedBy(B, A) if equal(B, A).
% 0.350 (1.000)
divides(B, A) if divides(B, A).
% 0.216 (0.617)
divides(B, A) if inv_dividedBy(B, A).
% 0.152 (0.434)
divides(C, A) if divides(B, A), inv_dividedBy(C, B).
% 0.096 (0.273)
divides(B, A) if equal(B, A).
% 0.094 (0.268)
divides(C, A) if inv_dividedBy(B, A), inv_dividedBy(C, B).
% 0.036 (0.103)
divides(C, A) if divides(B, A), divides(C, B).
% 0.028 (0.080)
divides(B, A) if inv_dividedBy(B, A).
% 0.022 (0.063)
divides(C, A) if inv_dividedBy(B, A), divides(C, B).
% 0.285 (1.000)
notRelated(C, A) if inv_notRelated(B, A), inv_notRelated(C, B).
% 0.213 (0.748)
notRelated(C, A) if notRelated(B, A), inv_notRelated(C, B).
% 0.142 (0.498)
notRelated(C, A) if inv_notRelated(B, A), notRelated(C, B).
% 0.106 (0.373)
notRelated(C, A) if notRelated(B, A), notRelated(C, B).
% 0.071 (0.249)
notRelated(C, A) if inv_divides(B, A), inv_notRelated(C, B).
% 0.046 (0.163)
notRelated(C, A) if dividedBy(B, A), inv_notRelated(C, B).
% 0.035 (0.124)
notRelated(C, A) if inv_divides(B, A), notRelated(C, B).
% 0.034 (0.119)
notRelated(B, A) if inv_notRelated(B, A).
% 0.023 (0.081)
notRelated(C, A) if dividedBy(B, A), notRelated(C, B).
% 0.017 (0.059)
notRelated(B, A) if notRelated(B, A).
% 0.332 (1.000)
inv_dividedBy(B, A) if divides(B, A).
% 0.282 (0.851)
inv_dividedBy(B, A) if inv_dividedBy(B, A).
% 0.160 (0.482)
inv_dividedBy(C, A) if divides(B, A), inv_dividedBy(C, B).
% 0.136 (0.411)
inv_dividedBy(C, A) if inv_dividedBy(B, A), inv_dividedBy(C, B).
% 0.054 (0.162)
inv_dividedBy(B, A) if equal(B, A).
% 0.017 (0.051)
inv_dividedBy(B, A) if inv_dividedBy(B, A).
% 0.340 (1.000)
inv_divides(C, A) if inv_divides(B, A), dividedBy(C, B).
% 0.243 (0.717)
inv_divides(C, A) if inv_divides(B, A), inv_divides(C, B).
% 0.165 (0.486)
inv_divides(B, A) if inv_divides(B, A).
% 0.100 (0.296)
inv_divides(B, A) if dividedBy(B, A).
% 0.072 (0.212)
inv_divides(B, A) if inv_divides(B, A).
% 0.032 (0.095)
inv_divides(B, A) if equal(B, A).
% 0.022 (0.063)
inv_divides(C, A) if dividedBy(B, A), dividedBy(C, B).
% 0.015 (0.045)
inv_divides(C, A) if dividedBy(B, A), inv_divides(C, B).
% 0.010 (0.031)
inv_divides(B, A) if dividedBy(B, A).
% 0.214 (1.000)
inv_notRelated(C, A) if notRelated(B, A), notRelated(C, B).
% 0.205 (0.954)
inv_notRelated(C, A) if notRelated(B, A), inv_notRelated(C, B).
% 0.145 (0.676)
inv_notRelated(C, A) if inv_notRelated(B, A), notRelated(C, B).
% 0.138 (0.645)
inv_notRelated(C, A) if inv_notRelated(B, A), inv_notRelated(C, B).
% 0.058 (0.270)
inv_notRelated(C, A) if dividedBy(B, A), notRelated(C, B).
% 0.055 (0.258)
inv_notRelated(C, A) if dividedBy(B, A), inv_notRelated(C, B).
% 0.051 (0.240)
inv_notRelated(C, A) if inv_divides(B, A), notRelated(C, B).
% 0.049 (0.229)
inv_notRelated(C, A) if inv_divides(B, A), inv_notRelated(C, B).
% 0.023 (0.109)
inv_notRelated(B, A) if notRelated(B, A).
% 0.022 (0.104)
inv_notRelated(B, A) if inv_notRelated(B, A).
