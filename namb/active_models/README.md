# active_models

Drop `.nam` files here. Everything in this directory is processed by the
`ModelPack: build (ch3 / ch8)` tasks:

    active_models/*.nam
        -> split_slimmable  -> build/split_models/*.nam
        -> nam2namb         -> build/namb_models/*.namb
        -> nambpack         -> build/modelpack.bin

Container models (`SlimmableContainer`, which is how every A2 model ships) are
split into their `_ch3` / `_ch8` submodels; plain WaveNet `.nam` files pass
straight through. The pack step then selects by channel count, so the same
directory feeds both the A2-Lite and A2-Full packs.

This is a working set, not bulk storage -- the intermediate directories are
wiped on every run so a model removed from here also leaves the pack.
