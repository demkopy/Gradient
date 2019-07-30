import py_expression_eval as prsr
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

parser = prsr.Parser()
formula = str(input("f(x,y) = "))


def func_temp(formula_):
    parsed = parser.parse(formula_)
    vars = parsed.variables()

    def spec_func(x, y):
        return round(parsed.evaluate({vars[0]: x, vars[1]: y}), 2)

    return spec_func


lb_x = float(input("Type left boundary for 'x': "))
rb_x = float(input("Type right boundary for 'x': "))

lb_y = float(input("Type left boundary for 'y': "))
rb_y = float(input("Type right boundary for 'y': "))

segs = 500
rng_x = np.linspace(lb_x, rb_x, segs)
rng_y = np.linspace(lb_y, rb_y, segs)

func = func_temp(formula)

xs, ys = np.meshgrid(rng_x, rng_y)
zs = np.array([[func(x, y) for x, y in zip(xs[i], ys[i])] for i in range(0, len(rng_x))])

plt.pcolormesh(xs, ys, zs, cmap="Greys")

# calculating derivatives


def replace_all(res):
    res = res.replace("sin", "sp.sin")
    res = res.replace("cos", "sp.cos")
    res = res.replace("tan", "sp.tan")
    res = res.replace("cot", "sp.cot")
    res = res.replace("exp", "sp.exp")
    res = res.replace("atan", "sp.atan")
    res = res.replace("^", "**")
    return res


x, y = sp.symbols("x y")

exec("nf = " + replace_all(formula))

deriv_x = sp.diff(nf, x)
deriv_y = sp.diff(nf, y)

# calculating minimum

g_step = 0.1
eps = 0.000001

plot = {x: [], y: []}

xx, yy = 5, 10
prev_f = func(xx, yy)
f_diff = 2 * eps

plot[x].append(xx)
plot[y].append(yy)

while f_diff > eps:
    print(str(xx) + " " + str(yy))
    dx = deriv_x.evalf(subs={x: xx})
    dy = deriv_y.evalf(subs={y: yy})

    temp_step = g_step
    while True:
        temp_xx = xx - dx * temp_step
        temp_yy = yy - dy * temp_step
        temp_f = func(temp_xx, temp_yy)
        if temp_f > prev_f:
            temp_step *= 0.5
        else:
            xx = temp_xx
            yy = temp_yy
            g_step = temp_step
            next_f = temp_f
            break

    f_diff = abs(prev_f - next_f)
    prev_f = next_f
    plot[x].append(xx)
    plot[y].append(yy)

print("x = " + str(xx) + " y = " + str(yy))

plt.plot(plot[x], plot[y], "r-")
plt.scatter(plot[x][-1], plot[y][-1], marker="x")

plt.show()
