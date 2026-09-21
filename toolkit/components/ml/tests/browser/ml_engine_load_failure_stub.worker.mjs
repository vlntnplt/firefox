/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

// Fails while the worker script loads, before any request is handled.
throw new Error("The worker script could not load");
