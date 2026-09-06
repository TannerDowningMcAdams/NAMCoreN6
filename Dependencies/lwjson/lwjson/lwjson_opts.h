#pragma once
// LwJSON configuration for the .nam streaming front end.
//
// lwjson_opt.h includes this file and every setting it does not find here keeps
// its upstream default. Only the streaming parser is built -- lwjson.c, the DOM
// half, is not vendored -- so nothing below touches token or DOM behaviour.

// Deepest nesting the .nam grammar reaches, plus room.
//
// A container bottoms out at 10: root > config > submodels > [i] > model >
// config > layers > [j] > activation > [k]. Every key on the path occupies a
// stack entry of its own, so the practical figure is roughly twice that; 32 is
// comfortably clear of it and costs 32 * sizeof(lwjson_stream_stack_t).
#define LWJSON_CFG_STREAM_STACK_SIZE 32

// Longest number token we will accept.
//
// The default is 32, which is thin: a round-tripped double runs to 24-odd
// characters before an exponent, and lwjson refuses the document rather than
// truncating, so a tight bound here is a model that will not import. Weights
// are the tokens this applies to and there are tens of thousands of them.
#define LWJSON_CFG_STREAM_PRIMITIVE_MAX_LEN 64

// Longest object key we can match on.
//
// Truncation here is silent -- a longer key is stored as its first N characters
// with no error -- and every routing decision in namb_writer.h is a key
// comparison, so a truncated key is a field read into the wrong slot. The
// longest the .nam grammar uses is "input_mixin_post_film" at 21; 48 leaves the
// margin the failure mode deserves.
#define LWJSON_CFG_STREAM_KEY_MAX_LEN 48

// Longest string value we keep whole.
//
// Model names run long -- "[AMP] Mes.BADLND-S 050W BOLD CRNCH Noon #09 - DI" is
// a real one -- and are sanitised down to 31 characters afterwards anyway. The
// default 256 is already generous; the parser chunks anything longer rather
// than failing, and DeriveName only ever sees the first chunk.
#define LWJSON_CFG_STREAM_STRING_MAX_LEN 256
