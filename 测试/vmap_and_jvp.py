import jax
import jax.numpy as jnp

# 多变量函数
def f(x):
    return jnp.array([jnp.sin(x[0]) + x[1], x[0] * x[1]])

x_1=jnp.array([1.,2.])
y_1=f(x_1)
print("f(x_1):\n",y_1)

vmap_f = jax.vmap(f)
x_2 = jnp.array([[0.0, 1.0], [1.0, 2.0]])  # 2个输入点
y_2 = vmap_f(x_2)  # 形状为 (2, 2)
print("vmap_f(x_2)\n",y_2)

def compute_vjp(x, cotangent):
    primals, vjp_fn = jax.vjp(f, x)
    gradients = vjp_fn(cotangent)
    return primals, gradients

vmap_vjp = jax.vmap(compute_vjp, in_axes=(0, 0))

primals, gradients = vmap_vjp(x_2, jnp.ones_like(y_2) )

print("函数值:\n", primals)
print("梯度:\n", gradients)

primals, gradients = compute_vjp(x_1, jnp.ones_like(y_1))

print("函数值:\n", primals)
print("梯度:\n", gradients)

def compute_vjp(x, cotangent):
    primals, vjp_fn = jax.vjp(vmap_f, x)
    gradients = vjp_fn(cotangent)
    return primals, gradients
primals, gradients = compute_vjp(x_2, jnp.ones_like(y_2) )
print("函数值:\n", primals)
print("梯度:\n", gradients)

