/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * JSON schema to llama.cpp GBNF grammar, after
 * examples/json_schema_to_grammar.py in llama.cpp.
 *
 * Supported: type object (properties, required, additionalProperties),
 * array (items, minItems, maxItems), string (minLength, maxLength), number,
 * integer, boolean, null, enum, const, and the empty schema. Anything else
 * throws: the grammar parser aborts the inference process on input it
 * rejects, so only grammars this module emits may reach it.
 */

const SPACE_RULE = '| " " | "\\n"{1,2} [ \\t]{0,20}';

/** @type {Record<string, { content: string, deps: string[] }>} */
const PRIMITIVE_RULES = {
  boolean: { content: '("true" | "false") space', deps: [] },
  "decimal-part": { content: "[0-9]{1,16}", deps: [] },
  "integral-part": { content: "[0] | [1-9] [0-9]{0,15}", deps: [] },
  number: {
    content:
      '("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
    deps: ["integral-part", "decimal-part"],
  },
  integer: {
    content: '("-"? integral-part) space',
    deps: ["integral-part"],
  },
  value: {
    content: "object | array | string | number | boolean | null",
    deps: ["object", "array", "string", "number", "boolean", "null"],
  },
  object: {
    content:
      '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
    deps: ["string", "value"],
  },
  array: {
    content: '"[" space ( value ("," space value)* )? "]" space',
    deps: ["value"],
  },
  char: {
    content:
      '[^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
    deps: [],
  },
  string: { content: '"\\"" char* "\\"" space', deps: ["char"] },
  null: { content: '"null" space', deps: [] },
};

const RESERVED_NAMES = new Set([
  "root",
  "dot",
  ...Object.keys(PRIMITIVE_RULES),
]);

const SUPPORTED_KEYWORDS = new Set([
  "type",
  "properties",
  "required",
  "additionalProperties",
  "items",
  "prefixItems",
  "anyOf",
  "oneOf",
  "definitions",
  "$defs",
  "minItems",
  "maxItems",
  "minLength",
  "maxLength",
  "enum",
  "const",
  "title",
  "description",
  "default",
  "examples",
  "$schema",
  "$id",
]);

const LITERAL_ESCAPES = { "\r": "\\r", "\n": "\\n", '"': '\\"', "\\": "\\\\" };

/** @param {string} literal */
function formatLiteral(literal) {
  return `"${literal.replace(/[\r\n"\\]/g, c => LITERAL_ESCAPES[c])}"`;
}

/**
 * @param {string} itemRule
 * @param {number} minItems
 * @param {number | undefined} maxItems
 * @param {string} [separatorRule]
 */
function buildRepetition(itemRule, minItems, maxItems, separatorRule) {
  if (maxItems === 0) {
    return "";
  }
  if (minItems === 0 && maxItems === 1) {
    return `${itemRule}?`;
  }
  if (!separatorRule) {
    if (minItems === 1 && maxItems === undefined) {
      return `${itemRule}+`;
    }
    if (minItems === 0 && maxItems === undefined) {
      return `${itemRule}*`;
    }
    return `${itemRule}{${minItems},${maxItems ?? ""}}`;
  }
  const result =
    itemRule +
    " " +
    buildRepetition(
      `(${separatorRule} ${itemRule})`,
      minItems > 0 ? minItems - 1 : 0,
      maxItems === undefined ? undefined : maxItems - 1
    );
  return minItems === 0 ? `(${result})?` : result;
}

class SchemaConverter {
  /** @type {Map<string, string>} */
  #rules = new Map([["space", SPACE_RULE]]);

  /**
   * @param {string} name
   * @param {string} rule
   * @returns {string} The registered rule name.
   */
  #addRule(name, rule) {
    const escaped = name.replace(/[^a-zA-Z0-9-]+/g, "-");
    let key = escaped;
    if (this.#rules.has(escaped) && this.#rules.get(escaped) !== rule) {
      let i = 0;
      while (
        this.#rules.has(`${escaped}${i}`) &&
        this.#rules.get(`${escaped}${i}`) !== rule
      ) {
        i++;
      }
      key = `${escaped}${i}`;
    }
    this.#rules.set(key, rule);
    return key;
  }

  /**
   * @param {string} name
   * @param {string} [primitive]
   */
  #addPrimitive(name, primitive = name) {
    const rule = PRIMITIVE_RULES[primitive];
    const key = this.#addRule(name, rule.content);
    for (const dep of rule.deps) {
      if (!this.#rules.has(dep)) {
        this.#addPrimitive(dep);
      }
    }
    return key;
  }

  /** @param {unknown} value */
  #constantRule(value) {
    return formatLiteral(JSON.stringify(value));
  }

  /**
   * @param {object} schema
   * @param {string} name
   * @returns {string} The rule name for the schema.
   */
  visit(schema, name) {
    if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
      throw new Error(`Unsupported JSON schema: ${JSON.stringify(schema)}`);
    }
    for (const keyword of Object.keys(schema)) {
      if (!SUPPORTED_KEYWORDS.has(keyword)) {
        throw new Error(`Unsupported JSON schema keyword: ${keyword}`);
      }
    }
    const type = schema.type;
    const ruleName = RESERVED_NAMES.has(name) ? `${name}-` : name || "root";
    const prefix = name ? `${name}-` : "";

    const alternatives = schema.oneOf ?? schema.anyOf;
    if (alternatives) {
      return this.#addRule(ruleName, this.#unionRule(alternatives, name));
    }
    if (Array.isArray(type)) {
      return this.#addRule(
        ruleName,
        this.#unionRule(
          type.map(t => ({ ...schema, type: t })),
          name
        )
      );
    }
    if (type !== undefined && typeof type !== "string") {
      throw new Error(`Unsupported JSON schema type: ${JSON.stringify(type)}`);
    }
    if ("const" in schema) {
      return this.#addRule(
        ruleName,
        `${this.#constantRule(schema.const)} space`
      );
    }
    if ("enum" in schema) {
      if (!Array.isArray(schema.enum) || !schema.enum.length) {
        throw new Error(
          "Unsupported JSON schema: enum must be a non-empty array"
        );
      }
      const literals = schema.enum.map(v => this.#constantRule(v));
      return this.#addRule(ruleName, `(${literals.join(" | ")}) space`);
    }
    if (
      (type === undefined || type === "object") &&
      ("properties" in schema ||
        ("additionalProperties" in schema &&
          schema.additionalProperties !== true))
    ) {
      const required = new Set(schema.required ?? []);
      const properties = Object.entries(schema.properties ?? {});
      return this.#addRule(
        ruleName,
        this.#objectRule(
          properties,
          required,
          name,
          schema.additionalProperties
        )
      );
    }
    if (
      (type === undefined || type === "array") &&
      ("items" in schema || "prefixItems" in schema)
    ) {
      const items = schema.items ?? schema.prefixItems;
      if (Array.isArray(items)) {
        throw new Error("Unsupported JSON schema: tuple items");
      }
      const itemRule = this.visit(items, `${prefix}item`);
      const repetition = buildRepetition(
        itemRule,
        schema.minItems ?? 0,
        schema.maxItems,
        '"," space'
      );
      return this.#addRule(ruleName, `"[" space ${repetition} "]" space`);
    }
    if (type === "string" && ("minLength" in schema || "maxLength" in schema)) {
      const charRule = this.#addPrimitive("char");
      const repetition = buildRepetition(
        charRule,
        schema.minLength ?? 0,
        schema.maxLength
      );
      return this.#addRule(ruleName, `"\\"" ${repetition} "\\"" space`);
    }
    if (type === "object" || !Object.keys(schema).length) {
      return this.#addRule(ruleName, this.#addPrimitive("object"));
    }
    if (type === undefined) {
      return this.#addRule(ruleName, this.#addPrimitive("value"));
    }
    if (!Object.hasOwn(PRIMITIVE_RULES, type)) {
      throw new Error(`Unsupported JSON schema type: ${type}`);
    }
    return this.#addPrimitive(ruleName === "root" ? "root" : type, type);
  }

  /**
   * @param {object[]} alternatives
   * @param {string} name
   */
  #unionRule(alternatives, name) {
    if (!Array.isArray(alternatives) || !alternatives.length) {
      throw new Error("Unsupported JSON schema: empty union");
    }
    const prefix = name ? `${name}-` : "alternative-";
    return alternatives
      .map((alt, i) => this.visit(alt, `${prefix}${i}`))
      .join(" | ");
  }

  /**
   * @param {Array<[string, object]>} properties
   * @param {Set<string>} required
   * @param {string} name
   * @param {unknown} additionalProperties
   */
  #objectRule(properties, required, name, additionalProperties) {
    const prefix = name ? `${name}-` : "";
    /** @type {Map<string, string>} */
    const kvRuleNames = new Map();
    for (const [propName, propSchema] of properties) {
      const propRule = this.visit(propSchema, `${prefix}${propName}`);
      kvRuleNames.set(
        propName,
        this.#addRule(
          `${prefix}${propName}-kv`,
          `${formatLiteral(JSON.stringify(propName))} space ":" space ${propRule}`
        )
      );
    }
    const requiredProps = properties
      .map(([k]) => k)
      .filter(k => required.has(k));
    const optionalProps = properties
      .map(([k]) => k)
      .filter(k => !required.has(k));

    if (additionalProperties !== undefined && additionalProperties !== false) {
      const subName = `${prefix}additional`;
      const valueRule =
        typeof additionalProperties === "object"
          ? this.visit(additionalProperties, `${subName}-value`)
          : this.#addPrimitive("value");
      if (properties.length) {
        throw new Error(
          "Unsupported JSON schema: additionalProperties alongside properties"
        );
      }
      const keyRule = this.#addPrimitive("string");
      kvRuleNames.set(
        "*",
        this.#addRule(`${subName}-kv`, `${keyRule} ":" space ${valueRule}`)
      );
      optionalProps.push("*");
    }

    let rule = '"{" space ';
    rule += requiredProps.map(k => kvRuleNames.get(k)).join(' "," space ');

    if (optionalProps.length) {
      rule += " (";
      if (requiredProps.length) {
        rule += ' "," space ( ';
      }
      /**
       * @param {string[]} keys
       * @param {boolean} firstIsOptional
       */
      const recursiveRefs = (keys, firstIsOptional) => {
        const [k, ...rest] = keys;
        const kvRuleName = kvRuleNames.get(k);
        const commaRef = `( "," space ${kvRuleName} )`;
        let result;
        if (firstIsOptional) {
          result = commaRef + (k === "*" ? "*" : "?");
        } else {
          result = kvRuleName + (k === "*" ? ` ${commaRef}*` : "");
        }
        if (rest.length) {
          result +=
            " " +
            this.#addRule(`${prefix}${k}-rest`, recursiveRefs(rest, true));
        }
        return result;
      };
      rule += optionalProps
        .map((_, i) => recursiveRefs(optionalProps.slice(i), false))
        .join(" | ");
      if (requiredProps.length) {
        rule += " )";
      }
      rule += " )?";
    }

    rule += ' "}" space';
    return rule;
  }

  format() {
    const names = [...this.#rules.keys()].sort();
    return names.map(name => `${name} ::= ${this.#rules.get(name)}`).join("\n");
  }
}

/**
 * @param {object} schema
 * @returns {string} A GBNF grammar whose root rule accepts exactly the JSON
 *   documents valid under the schema subset this module supports.
 */
export function jsonSchemaToGrammar(schema) {
  const converter = new SchemaConverter();
  converter.visit(schema, "");
  return converter.format();
}

/** Any JSON value. */
export const JSON_GRAMMAR = jsonSchemaToGrammar({});
