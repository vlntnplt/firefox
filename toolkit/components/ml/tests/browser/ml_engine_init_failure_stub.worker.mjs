/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

import { _lazyForTestMocking } from "chrome://global/content/ml/MLEngine.worker.mjs";

// A DOMException, as file I/O throws: not one of the errors PromiseWorker
// rebuilds on the receiving side, so it crosses the worker as bare data.
_lazyForTestMocking.getBackend = async () => {
  throw new DOMException(
    "The model file went away while loading",
    "NotFoundError"
  );
};
