// Driver invoked by tests/test_js_parity.py: loads the exported weights
// JSON and a batch of raw hands, runs demo/forward_pass.js's classify()
// on each, writes the predictions back out for the Python side to compare
// against forward_numpy.
const fs = require("fs");
const path = require("path");
const { classify } = require(path.join(__dirname, "..", "demo", "forward_pass.js"));

const [, , weightsPath, inputPath, outputPath] = process.argv;

const net = JSON.parse(fs.readFileSync(weightsPath, "utf8"));
const hands = JSON.parse(fs.readFileSync(inputPath, "utf8"));

const results = hands.map((raw) => {
  const { classIndex, logits } = classify(net, raw);
  return { classIndex, logits };
});

fs.writeFileSync(outputPath, JSON.stringify(results));
