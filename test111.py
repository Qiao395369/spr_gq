import jax
import jax.numpy as jnp

# 假设 f(x, y) 是一个标量函数，x 和 y 的形状为 (3,)。
def f(p,x, y):

    return jnp.sum(x)+jnp.sum(y)   # 示例标量函数

# 第一步：对 y 进行批处理，适配形状为 (ne, 3)。
# 输出形状为 (ne,)
f_vmapped_y = jax.vmap(f, in_axes=(None, 0))  # 针对 y 的第一个维度批量化

# 第二步：对 x 进行批处理，适配形状为 (na, 3) 和 (ne, 3)。
# 输出形状为 (na, ne)
f_vmapped_W = jax.vmap(f_vmapped_y, in_axes=(0, 0))  # 针对 x 的第一个维度批量化

# 第三步：对 x 和 y 进行批处理，适配形状为 (W, na, 3) 和 (W, ne, 3)。
# 输出形状为 (W, na, ne)
# f_vmapped_W = jax.vmap(f_vmapped_x_y, in_axes=(0, 0))  # 针对 W 维度批量化

# 第四步：对 B 批量化，适配形状为 (W, na, 3) 和 (W, B, ne, 3)。
# 输出形状为 (W, B)
def batched_f(f):
    return lambda p,xa,xe: jax.vmap(lambda xa_, xe_: jax.vmap(lambda xe__: f(p,xa_, xe__),in_axes=0)(xe_),in_axes=(0, 0))(xa, xe)

x=jnp.ones((4,3))
y=jnp.ones((2,3))
xx=jnp.ones((8,4,3))
yy=jnp.ones((8,6,2,3))

print(f(1,x,y))
# print(f_vmapped_W(xx,yy).shape)
# print(f_vmapped_W(xx,yy))

print(batched_f(f)(1,xx,yy).shape)
print(batched_f(f)(1,xx,yy))