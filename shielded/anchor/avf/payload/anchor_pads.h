/* Dealt pads inside the pVM (shielded/dealer/PLAN.md): what the payload hands
 * libengine.so so the engine can ask the owner app for ledger windows. All
 * pointers stay valid for the engine's lifetime; the secrets never leave the
 * VM (the transport secret signs requests, the pad secret opened the seed). */
#ifndef ANCHOR_PADS_H
#define ANCHOR_PADS_H
#include <stdint.h>
typedef struct {
    const uint8_t *transport_sk;   /* 64 bytes, TweetNaCl Ed25519 secret (signs PADWIN requests) */
    const uint8_t *ledger_pk;      /* 32 bytes, the relay's ledger key (verifies PADWIN replies) */
    const char    *name;           /* this host's tunnel name (part of every signed request) */
    const char    *seed_id_hex;    /* 32 hex, the seed the platform issued to this pVM */
} anchor_pads;
#endif
