# lwjson (vendored)

Streaming JSON parser, used by `namb/namb_writer.h` to convert a `.nam` without
building a DOM.

Vendored rather than submoduled for the same reason `nlohmann/` is: NAMCoreN6
has to configure, build and test on its own, and `namb/tools/CMakeLists.txt`
cannot reach a submodule of whatever superproject happens to contain it.

| | |
|---|---|
| Upstream | <https://github.com/MaJerle/lwjson> |
| Commit | `79a8ad43f29a0ac8696c2618a5721ab08c92a98e` (`develop`, post-v1.6.1) |
| Licence | MIT, Copyright (c) 2025 Tilen Majerle -- see `LICENSE` |

## What is here

Only the streaming half. `lwjson.c` -- the DOM parser and its token pool -- is
deliberately absent: building a DOM is the thing this dependency exists to
avoid, and leaving it out means it cannot be reached for by accident.

```
lwjson_stream.c          the parser
lwjson/lwjson.h          its API, plus DOM declarations we do not link
lwjson/lwjson_opt.h      upstream defaults
lwjson/lwjson_opts.h     OURS -- the only file here that is not upstream
lwjson/lwjson_utils.h    included by lwjson_stream.c
lwjson/lwjson_serializer.h   included by lwjson_utils.h
```

`lwjson_opts.h` is the sanctioned override hook: `lwjson_opt.h` includes it and
falls back to its own defaults for anything it does not define. Configuration
belongs there and nowhere else.

## Updating

Re-copy the files above from upstream, leaving `lwjson_opts.h` alone, then run
`ctest` from the host tools build. `writer_test` pins the converter's output
against goldens byte for byte, so a change in number handling shows up as a
test failure rather than as models that load slightly wrong.
