// Pure computational core of the demo: the same 17-dim card-order-invariant
// encoding (poker_handnet/encoding.py) and the same dense matmul+bias+ReLU
// forward pass (dag.py's CompiledNet.forward_numpy), re-expressed in plain
// JS with no DOM dependency. Loaded two ways:
//  - inlined verbatim into demo/index.html by scripts/build_demo.py, where
//    `typeof module` is undefined and these stay plain globals;
//  - required() from tests/test_js_parity.py via Node, to check this file
//    produces bit-for-bit the same predictions as forward_numpy on random
//    sample hands.

function encode(raw) {
  // raw: [S1,C1,S2,C2,S3,C3,S4,C4,S5,C5], suit in 1..4, rank in 1..13 (ace=1)
  const suitCount = [0, 0, 0, 0];
  const rankCount = new Array(13).fill(0);
  for (let i = 0; i < 5; i++) {
    const suit = raw[2 * i];
    const rank = raw[2 * i + 1];
    suitCount[suit - 1] += 1;
    rankCount[rank - 1] += 1;
  }
  return suitCount.concat(rankCount);
}

function forwardWithActivations(netJson, x) {
  // activations[0] is the encoded input; activations[i+1] is layer i's
  // output. Kept as one pass so the demo's live per-neuron display can
  // never drift from the value actually used by the next layer.
  const activations = [x.slice()];
  let h = x;
  for (const layer of netJson.layers) {
    const out = new Array(layer.W.length);
    for (let r = 0; r < layer.W.length; r++) {
      let acc = layer.b[r];
      const row = layer.W[r];
      for (let c = 0; c < row.length; c++) {
        acc += row[c] * h[c];
      }
      out[r] = layer.activation === "relu" ? Math.max(0, acc) : acc;
    }
    h = out;
    activations.push(h);
  }
  return activations;
}

function forward(netJson, x) {
  const activations = forwardWithActivations(netJson, x);
  return activations[activations.length - 1];
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > arr[best]) best = i;
  }
  return best;
}

function classify(netJson, raw) {
  const x = encode(raw);
  const activations = forwardWithActivations(netJson, x);
  const logits = activations[activations.length - 1];
  return { classIndex: argmax(logits), logits, encoded: x, activations };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { encode, forward, forwardWithActivations, argmax, classify };
}
