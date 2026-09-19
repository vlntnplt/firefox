/**
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

import { createEngine } from "chrome://global/content/ml/EngineProcess.sys.mjs";
import {
  OAUTH_CLIENT_ID,
  SCOPE_PROFILE_UID,
  SCOPE_SMART_WINDOW,
} from "resource://gre/modules/FxAccountsCommon.sys.mjs";
import { XPCOMUtils } from "resource://gre/modules/XPCOMUtils.sys.mjs";

const lazy = XPCOMUtils.declareLazy({
  getFxAccountsSingleton: "resource://gre/modules/FxAccounts.sys.mjs",
  mlAttributionForModelFeature:
    "moz-src:///browser/components/aiwindow/models/Utils.sys.mjs",
});

const APIKEY_PREF = "browser.smartwindow.apiKey";
const ENDPOINT_PREF = "browser.smartwindow.endpoint";
const CUSTOM_ENDPOINT_PREF = "browser.smartwindow.customEndpoint";
const CUSTOM_MODEL_CHOICE_ID = "0";
const MLPA_ENDPOINT_NAME = "mlpa";
const CUSTOM_ENDPOINT_NAME = "custom";

/**
 * The default endpoint used for preset models
 */
const DEFAULT_ENDPOINT = "https://mlpa-prod-prod-mozilla.freetls.fastly.net/v1";

/**
 * Transport for AI Window LLM calls against an OpenAI-style backend.
 */
export class openAIEngine {
  /**
   * Exposing createEngine for testing purposes.
   */
  static _createEngine = createEngine;

  /**
   * The model config (a MODEL_FEATURES value) this instance was built for.
   *
   * @type {string | null}
   */
  feature = null;

  /**
   * The ML feature the shared endpoint engine attributes this instance's runs
   * to, and the activity they carry unless a run names another one.
   *
   * @type {string | null}
   */
  featureId = null;

  /** @type {string | null} */
  activity = null;

  /**
   * Resolved model name for LLM inference, sent with each request.
   *
   * @type {string | null}
   */
  model = null;

  /**
   * Service type sent with each request ("ai", "memories", "agent").
   *
   * @type {string | null}
   */
  #serviceType = null;

  /**
   * Purpose header sent with each request, used by the backend for routing.
   *
   * @type {string | null}
   */
  #purpose = null;

  /**
   * Base URL for this engine instance. Resolved during build time from
   * the selected model choice.
   *
   * @type {string | null}
   */
  #baseURL = null;

  /**
   * Checks whether a custom endpoint is configured via pref.
   *
   * @returns {boolean} True if the endpoint pref has a user-set value.
   */
  static hasCustomEndpoint() {
    return Services.prefs.prefHasUserValue(CUSTOM_ENDPOINT_PREF);
  }

  /**
   * Whether the current engine instance uses a custom model endpoint.
   *
   * @returns {boolean}
   */
  get isCustomEndpoint() {
    return this.#baseURL !== null && this.#baseURL !== openAIEngine.endpoint;
  }

  /**
   * The engine every call to an endpoint shares. The model, purpose and
   * service type travel with each request, so nothing else distinguishes
   * calls at the engine level.
   *
   * @param {string} baseURL
   * @returns {{ engineId: string, config: string }}
   */
  static engineIdentity(baseURL) {
    const endpoint =
      baseURL === openAIEngine.endpoint
        ? MLPA_ENDPOINT_NAME
        : CUSTOM_ENDPOINT_NAME;
    return {
      engineId: `sw-${endpoint}-completions`,
      config: `${endpoint}-completions`,
    };
  }

  /**
   * Returns the endpoint and API key for a model choice.
   *
   * @param {string} [modelChoiceId] - Selected model choice id
   * @returns {{baseURL: string, apiKey: string}}
   * @throws {Error} If the custom model choice is selected but not configured.
   */
  static resolveEndpointConfig(modelChoiceId) {
    if (modelChoiceId === CUSTOM_MODEL_CHOICE_ID) {
      const baseURL = Services.prefs.getStringPref(CUSTOM_ENDPOINT_PREF, "");
      if (!baseURL) {
        throw new Error("Custom model choice selected but not configured");
      }
      return {
        baseURL,
        apiKey: Services.prefs.getStringPref(APIKEY_PREF, ""),
      };
    }
    return { baseURL: openAIEngine.endpoint, apiKey: "" };
  }

  /**
   * Builds an openAIEngine instance.
   *
   * @param {object} options
   * @param {string} options.model
   * @param {string} options.serviceType
   * @param {string} options.purpose
   * @param {string|null} [options.flowId]
   * @param {string} options.feature
   * @param {string} [options.baseURL] - Endpoint base URL
   * @param {string} [options.apiKey] - API key for the endpoint
   * @returns {Promise<openAIEngine>}
   */
  static async build({
    model,
    serviceType,
    purpose,
    flowId,
    feature,
    baseURL = openAIEngine.endpoint,
    apiKey = "",
  }) {
    const engine = new openAIEngine();
    engine.feature = feature;
    if (feature) {
      const attribution = lazy.mlAttributionForModelFeature(feature);
      engine.featureId = attribution.featureId;
      engine.activity = attribution.activity;
    }
    engine.model = model;
    engine.#serviceType = serviceType;
    engine.#purpose = purpose;
    engine.#baseURL = baseURL;
    engine.engineInstance = await openAIEngine.#createOpenAIEngine({
      serviceType,
      purpose,
      flowId,
      featureId: engine.featureId,
      baseURL,
      apiKey,
    });
    return engine;
  }

  /**
   * Retrieves the Firefox account token
   *
   * @returns {Promise<string|null>}   The Firefox account token (string) or null
   */
  static async getFxAccountToken() {
    try {
      const fxAccounts = lazy.getFxAccountsSingleton();
      return await fxAccounts.getOAuthToken({
        scope: [SCOPE_SMART_WINDOW, SCOPE_PROFILE_UID],
        client_id: OAUTH_CLIENT_ID,
      });
    } catch (error) {
      console.warn("Error obtaining FxA token:", error);
      return null;
    }
  }

  /**
   * MLPA returns 429 for several sub-conditions (budget, QPS, upstream limit);
   * callers should back off the same way regardless of the sub-code.
   *
   * @param {Error} error
   * @returns {boolean}
   */
  static is429Error(error) {
    if (!error) {
      return false;
    }
    return error.status === 429 || !!error.message?.includes("429 status code");
  }

  /**
   * Checks if an error is transient and worth retrying: rate limits (429),
   * timeouts (408), conflicts (409), server errors (>= 500), and network-layer
   * failures. Deterministic errors (400/401/403/404/422) are NOT retryable.
   *
   * @param {Error} error  The error to check
   * @returns {boolean}    True if the error is transient and worth retrying
   */
  static isRetryableError(error) {
    if (!error) {
      return false;
    }
    if (this.is429Error(error)) {
      return true;
    }
    const isRetryableStatus = status =>
      status === 408 || status === 409 || (status >= 500 && status <= 599);
    if (typeof error.status === "number" && isRetryableStatus(error.status)) {
      return true;
    }
    // Some backends only encode the status in the message text, e.g.
    // "... 503 status code ...".
    const messageStatus = error.message?.match(/(\d{3}) status code/)?.[1];
    if (messageStatus && isRetryableStatus(Number(messageStatus))) {
      return true;
    }
    // Network-layer failures expose no HTTP status; in Gecko they surface as
    // NS_ERROR_* names or generic network/timeout messages.
    const text = `${error.name ?? ""} ${error.message ?? ""}`;
    return /NS_ERROR_(NET_|CONNECTION|PROXY)|NetworkError|connection (refused|reset)|timed? ?out/i.test(
      text
    );
  }

  /**
   * Creates (or reuses) the engine for an endpoint.
   *
   * @param {object} options
   * @param {string} options.serviceType  Default service type ("ai", "memories", "agent")
   * @param {string} options.purpose      Default purpose header
   * @param {string | null} [options.flowId]   Flow ID for correlating frontend and backend telemetry
   * @param {string | null} [options.featureId] - The ML feature id runs are attributed to
   * @param {string} options.baseURL - The endpoint base URL
   * @param {string} options.apiKey - The API key for the endpoint
   * @returns {Promise<object>} - The configured engine instance
   */
  static async #createOpenAIEngine({
    serviceType,
    purpose,
    flowId = null,
    featureId = null,
    baseURL,
    apiKey,
  }) {
    const extraHeadersPref = Services.prefs.getStringPref(
      "browser.smartwindow.extraHeaders",
      "{}"
    );
    let extraHeaders = {};
    try {
      extraHeaders = JSON.parse(extraHeadersPref);
    } catch (e) {
      console.error("Failed to parse extra headers from prefs:", e);
      Services.prefs.clearUserPref("browser.smartwindow.extraHeaders");
    }

    try {
      const engineInstance = await openAIEngine._createEngine({
        apiKey,
        backend: "openai",
        baseURL,
        ...openAIEngine.engineIdentity(baseURL),
        featureId,
        flowId,
        modelRevision: "main",
        taskName: "text-generation",
        serviceType,
        purpose,
        extraHeaders,
      });
      return engineInstance;
    } catch (error) {
      console.error("Failed to create OpenAI engine:", error);
      throw error;
    }
  }

  /**
   * Wrapper around engine.run to send message to the LLM
   * Will eventually use `usage` from the LiteLLM API response for token telemetry
   *
   * @param {Map<string, any>} content  OpenAI formatted messages to be sent to the LLM;
   *   `activity` names the run's activity instead of this instance's default
   * @returns {object}                  LLM response
   */
  async run(content) {
    const { activity = this.activity, ...request } = content;
    return await this._runWithAuth(request, activity);
  }

  /**
   * The request as the shared endpoint engine expects it: with the model,
   * service type and purpose this instance was built for.
   *
   * @param {object} content
   * @returns {object}
   */
  #request(content) {
    return {
      ...content,
      modelId: this.model,
      serviceType: this.#serviceType,
      purpose: this.#purpose,
    };
  }

  /**
   * Helper method to handle 401 authentication errors and retry with new token.
   *
   * @param {Map<string, any>} content  OpenAI formatted messages to be sent to the LLM
   * @param {string | null} activity    The run's activity for ML telemetry
   * @returns {object}                  LLM response
   */
  async _runWithAuth(content, activity = this.activity) {
    try {
      return await this.engineInstance.run(this.#request(content), {
        activity,
      });
    } catch (ex) {
      // Skip the token retry flow when using a custom endpoint,
      // as the retry logic only applies to FxAccounts tokens.
      if (!this._is401Error(ex) || this.isCustomEndpoint) {
        throw ex;
      }

      console.warn(
        "LLM request returned a 401 - revoking our token and retrying"
      );

      const fxAccounts = lazy.getFxAccountsSingleton();
      const oldToken = content.fxAccountToken;
      if (oldToken) {
        await fxAccounts.removeCachedOAuthToken({ token: oldToken });
      }

      const newToken = await openAIEngine.getFxAccountToken();
      const updatedContent = { ...content, fxAccountToken: newToken };

      try {
        return await this.engineInstance.run(this.#request(updatedContent), {
          activity,
        });
      } catch (retryEx) {
        if (!this._is401Error(retryEx)) {
          throw retryEx;
        }

        console.warn(
          "Retry LLM request still returned a 401 - revoking our token and failing"
        );

        if (newToken) {
          await fxAccounts.removeCachedOAuthToken({ token: newToken });
        }

        throw retryEx;
      }
    }
  }

  /**
   * Checks if an error is a 401 authentication error.
   *
   * @param {Error} error  The error to check
   * @returns {boolean}    True if the error is a 401 error
   * @private
   */
  _is401Error(error) {
    if (!error) {
      return false;
    }

    return error.status === 401 || error.message?.includes("401 status code");
  }

  /**
   * Helper async generator to handle 401 authentication errors and retry with new token for streaming requests.
   *
   * @param {Map<string, any>} options  OpenAI formatted messages with streaming and tooling options to be sent to the LLM
   * @yields {object}                   LLM streaming response chunks
   */
  async *_runWithGeneratorAuth(options) {
    // AbortSignal cannot be cloned via postMessage (structured clone algorithm).
    const { signal, activity = this.activity, ...engineOptions } = options;
    try {
      const generator = this.engineInstance.runWithGenerator(
        this.#request(engineOptions),
        { activity }
      );
      for await (const chunk of generator) {
        if (signal?.aborted) {
          return;
        }
        yield chunk;
      }
    } catch (ex) {
      // Skip the token retry flow when using a custom endpoint,
      // as the retry logic only applies to FxAccounts tokens.
      if (!this._is401Error(ex) || this.isCustomEndpoint) {
        throw ex;
      }

      console.warn(
        "LLM streaming request returned a 401 - revoking our token and retrying"
      );

      const fxAccounts = lazy.getFxAccountsSingleton();
      const oldToken = options.fxAccountToken;
      if (oldToken) {
        await fxAccounts.removeCachedOAuthToken({ token: oldToken });
      }

      const newToken = await openAIEngine.getFxAccountToken();
      const updatedOptions = { ...engineOptions, fxAccountToken: newToken };

      try {
        const generator = this.engineInstance.runWithGenerator(
          this.#request(updatedOptions),
          { activity }
        );
        for await (const chunk of generator) {
          if (signal?.aborted) {
            return;
          }
          yield chunk;
        }
      } catch (retryEx) {
        if (!this._is401Error(retryEx)) {
          throw retryEx;
        }

        console.warn(
          "Retry LLM streaming request still returned a 401 - revoking our token and failing"
        );

        if (newToken) {
          await fxAccounts.removeCachedOAuthToken({ token: newToken });
        }

        throw retryEx;
      }
    }
  }

  /**
   * Wrapper around engine.runWithGenerator to send message to the LLM
   * Will eventually use `usage` from the LiteLLM API response for token telemetry
   *
   * @param {Map<string, any>} options  OpenAI formatted messages with streaming and tooling options to be sent to the LLM;
   *   `activity` names the run's activity instead of this instance's default
   * @returns {AsyncGenerator}          LLM streaming response
   */
  runWithGenerator(options) {
    return this._runWithGeneratorAuth(options);
  }
}

XPCOMUtils.defineLazyPreferenceGetter(
  openAIEngine,
  "endpoint",
  ENDPOINT_PREF,
  DEFAULT_ENDPOINT
);

XPCOMUtils.defineLazyPreferenceGetter(openAIEngine, "apiKey", APIKEY_PREF, "");
