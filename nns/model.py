def make_sequential(layers):
    layers = list(layers)

    def forward(x):
        caches = []
        for layer in layers:
            x, c = layer["forward"](x)
            caches.append(c)
        return x, caches

    def backward(dout, caches):
        grads_list = [None] * len(layers)
        for i in reversed(range(len(layers))):
            dout, grads_list[i] = layers[i]["backward"](dout, caches[i])
        return dout, grads_list

    return {
        "forward": forward,
        "backward": backward,
        "params": [layer["params"] for layer in layers],
    }


def forward_backward(model, loss_fn, x, y):
    logits, caches = model["forward"](x)
    loss, d_logits = loss_fn(logits, y)
    _, param_grads = model["backward"](d_logits, caches)
    return float(loss), param_grads
