/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

const { jsonSchemaToGrammar, JSON_GRAMMAR } = ChromeUtils.importESModule(
  "moz-src:///toolkit/components/ml/textgeneration/JSONSchemaGrammar.sys.mjs"
);

// Expectations from llama.cpp tests/test-json-schema-to-grammar.cpp at the
// vendored revision, whitespace trimmed per line.
const UPSTREAM_CASES = [
  {
    name: "string",
    schema: { type: "string" },
    grammar: [
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'root ::= "\\"" char* "\\"" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "string w/ min length 1",
    schema: { type: "string", minLength: 1 },
    grammar: [
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'root ::= "\\"" char+ "\\"" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "string w/ min length 3",
    schema: { type: "string", minLength: 3 },
    grammar: [
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'root ::= "\\"" char{3,} "\\"" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "string w/ max length",
    schema: { type: "string", maxLength: 3 },
    grammar: [
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'root ::= "\\"" char{0,3} "\\"" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "string w/ min & max length",
    schema: { type: "string", minLength: 1, maxLength: 4 },
    grammar: [
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'root ::= "\\"" char{1,4} "\\"" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "boolean",
    schema: { type: "boolean" },
    grammar: [
      'root ::= ("true" | "false") space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "integer",
    schema: { type: "integer" },
    grammar: [
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      'root ::= ("-"? integral-part) space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "string const",
    schema: { const: "foo" },
    grammar: [
      'root ::= "\\"foo\\"" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "non-string const",
    schema: { const: 123 },
    grammar: [
      'root ::= "123" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "non-string enum",
    schema: { enum: ["red", "amber", "green", null, 42, ["foo"]] },
    grammar: [
      'root ::= ("\\"red\\"" | "\\"amber\\"" | "\\"green\\"" | "null" | "42" | "[\\"foo\\"]") space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "string array",
    schema: { type: "array", prefixItems: { type: "string" } },
    grammar: [
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'root ::= "[" space (string ("," space string)*)? "]" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
    ].join("\n"),
  },
  {
    name: "nullable string array",
    schema: { type: ["array", "null"], prefixItems: { type: "string" } },
    grammar: [
      'alternative-0 ::= "[" space (string ("," space string)*)? "]" space',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'null ::= "null" space',
      "root ::= alternative-0 | null",
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
    ].join("\n"),
  },
  {
    name: "array with empty items",
    schema: { type: "array", items: {} },
    grammar: [
      'array ::= "[" space ( value ("," space value)* )? "]" space',
      'boolean ::= ("true" | "false") space',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      "decimal-part ::= [0-9]{1,16}",
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      "item ::= object",
      'null ::= "null" space',
      'number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
      'object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
      'root ::= "[" space (item ("," space item)*)? "]" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
      "value ::= object | array | string | number | boolean | null",
    ].join("\n"),
  },
  {
    name: "array with empty items and prefixItems",
    schema: { type: "array", items: {}, prefixItems: { type: "string" } },
    grammar: [
      'array ::= "[" space ( value ("," space value)* )? "]" space',
      'boolean ::= ("true" | "false") space',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      "decimal-part ::= [0-9]{1,16}",
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      "item ::= object",
      'null ::= "null" space',
      'number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
      'object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
      'root ::= "[" space (item ("," space item)*)? "]" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
      "value ::= object | array | string | number | boolean | null",
    ].join("\n"),
  },
  {
    name: "number",
    schema: { type: "number" },
    grammar: [
      "decimal-part ::= [0-9]{1,16}",
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      'root ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "minItems",
    schema: { items: { type: "boolean" }, minItems: 2 },
    grammar: [
      'boolean ::= ("true" | "false") space',
      'root ::= "[" space boolean ("," space boolean)+ "]" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "maxItems 0",
    schema: { items: { type: "boolean" }, maxItems: 0 },
    grammar: [
      'boolean ::= ("true" | "false") space',
      'root ::= "[" space  "]" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "maxItems 1",
    schema: { items: { type: "boolean" }, maxItems: 1 },
    grammar: [
      'boolean ::= ("true" | "false") space',
      'root ::= "[" space boolean? "]" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "maxItems 2",
    schema: { items: { type: "boolean" }, maxItems: 2 },
    grammar: [
      'boolean ::= ("true" | "false") space',
      'root ::= "[" space (boolean ("," space boolean)?)? "]" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "min + maxItems",
    schema: {
      items: { type: ["number", "integer"] },
      minItems: 3,
      maxItems: 5,
    },
    grammar: [
      "decimal-part ::= [0-9]{1,16}",
      'integer ::= ("-"? integral-part) space',
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      "item ::= number | integer",
      'number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
      'root ::= "[" space item ("," space item){2,4} "]" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "required props in original order",
    schema: {
      type: "object",
      properties: {
        b: { type: "string" },
        c: { type: "string" },
        a: { type: "string" },
      },
      required: ["a", "b", "c"],
      additionalProperties: false,
      definitions: {},
    },
    grammar: [
      'a-kv ::= "\\"a\\"" space ":" space string',
      'b-kv ::= "\\"b\\"" space ":" space string',
      'c-kv ::= "\\"c\\"" space ":" space string',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'root ::= "{" space b-kv "," space c-kv "," space a-kv "}" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
    ].join("\n"),
  },
  {
    name: "1 optional prop",
    schema: {
      properties: { a: { type: "string" } },
      additionalProperties: false,
    },
    grammar: [
      'a-kv ::= "\\"a\\"" space ":" space string',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'root ::= "{" space  (a-kv )? "}" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
    ].join("\n"),
  },
  {
    name: "N optional props",
    schema: {
      properties: {
        a: { type: "string" },
        b: { type: "string" },
        c: { type: "string" },
      },
      additionalProperties: false,
    },
    grammar: [
      'a-kv ::= "\\"a\\"" space ":" space string',
      'a-rest ::= ( "," space b-kv )? b-rest',
      'b-kv ::= "\\"b\\"" space ":" space string',
      'b-rest ::= ( "," space c-kv )?',
      'c-kv ::= "\\"c\\"" space ":" space string',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'root ::= "{" space  (a-kv a-rest | b-kv b-rest | c-kv )? "}" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
    ].join("\n"),
  },
  {
    name: "required + optional props each in original order",
    schema: {
      properties: {
        b: { type: "string" },
        a: { type: "string" },
        d: { type: "string" },
        c: { type: "string" },
      },
      required: ["a", "b"],
      additionalProperties: false,
    },
    grammar: [
      'a-kv ::= "\\"a\\"" space ":" space string',
      'b-kv ::= "\\"b\\"" space ":" space string',
      'c-kv ::= "\\"c\\"" space ":" space string',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      'd-kv ::= "\\"d\\"" space ":" space string',
      'd-rest ::= ( "," space c-kv )?',
      'root ::= "{" space b-kv "," space a-kv ( "," space ( d-kv d-rest | c-kv ) )? "}" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
    ].join("\n"),
  },
  {
    name: "additional props",
    schema: {
      type: "object",
      additionalProperties: { type: "array", items: { type: "number" } },
    },
    grammar: [
      'additional-kv ::= string ":" space additional-value',
      'additional-value ::= "[" space (number ("," space number)*)? "]" space',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      "decimal-part ::= [0-9]{1,16}",
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      'number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
      'root ::= "{" space  (additional-kv ( "," space additional-kv )* )? "}" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
    ].join("\n"),
  },
  {
    name: "additional props (true)",
    schema: { type: "object", additionalProperties: true },
    grammar: [
      'array ::= "[" space ( value ("," space value)* )? "]" space',
      'boolean ::= ("true" | "false") space',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      "decimal-part ::= [0-9]{1,16}",
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      'null ::= "null" space',
      'number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
      'object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
      "root ::= object",
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
      "value ::= object | array | string | number | boolean | null",
    ].join("\n"),
  },
  {
    name: "additional props (implicit)",
    schema: { type: "object" },
    grammar: [
      'array ::= "[" space ( value ("," space value)* )? "]" space',
      'boolean ::= ("true" | "false") space',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      "decimal-part ::= [0-9]{1,16}",
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      'null ::= "null" space',
      'number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
      'object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
      "root ::= object",
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
      "value ::= object | array | string | number | boolean | null",
    ].join("\n"),
  },
  {
    name: "empty w/o additional props",
    schema: { type: "object", additionalProperties: false },
    grammar: [
      'root ::= "{" space  "}" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "conflicting names",
    schema: {
      type: "object",
      properties: {
        number: {
          type: "object",
          properties: {
            number: {
              type: "object",
              properties: { root: { type: "number" } },
              required: ["root"],
              additionalProperties: false,
            },
          },
          required: ["number"],
          additionalProperties: false,
        },
      },
      required: ["number"],
      additionalProperties: false,
      definitions: {},
    },
    grammar: [
      "decimal-part ::= [0-9]{1,16}",
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      'number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
      'number- ::= "{" space number-number-kv "}" space',
      'number-kv ::= "\\"number\\"" space ":" space number-',
      'number-number ::= "{" space number-number-root-kv "}" space',
      'number-number-kv ::= "\\"number\\"" space ":" space number-number',
      'number-number-root-kv ::= "\\"root\\"" space ":" space number',
      'root ::= "{" space number-kv "}" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
  {
    name: "description only (no type) treated as unconstrained",
    schema: {
      description:
        "The 0-based index of the last line to be retrieved (inclusive). If None, read until the end of the file.",
    },
    grammar: [
      'array ::= "[" space ( value ("," space value)* )? "]" space',
      'boolean ::= ("true" | "false") space',
      'char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
      "decimal-part ::= [0-9]{1,16}",
      "integral-part ::= [0] | [1-9] [0-9]{0,15}",
      'null ::= "null" space',
      'number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
      'object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
      "root ::= value",
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
      'string ::= "\\"" char* "\\"" space',
      "value ::= object | array | string | number | boolean | null",
    ].join("\n"),
  },
  {
    name: "literal string with escapes",
    schema: {
      properties: {
        code: {
          const: ' \r \n " \\ ',
          description: "Generated code",
          title: "Code",
          type: "string",
        },
      },
      required: ["code"],
      title: "DecoderResponse",
      type: "object",
    },
    grammar: [
      'code ::= "\\" \\\\r \\\\n \\\\\\" \\\\\\\\ \\"" space',
      'code-kv ::= "\\"code\\"" space ":" space code',
      'root ::= "{" space code-kv "}" space',
      'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
    ].join("\n"),
  },
];

const MEMORIES_MERGE_SCHEMA = {
  type: "array",
  items: {
    type: "object",
    additionalProperties: false,
    required: ["component_statements", "new_reasoning", "new_statement"],
    properties: {
      component_statements: {
        type: "array",
        minItems: 2,
        items: { type: "string" },
      },
      new_reasoning: { type: "string" },
      new_statement: { type: "string", maxLength: 100 },
    },
  },
};

add_task(function test_upstream_cases() {
  for (const { name, schema, grammar } of UPSTREAM_CASES) {
    Assert.equal(jsonSchemaToGrammar(schema), grammar, name);
  }
});

add_task(function test_memories_schemas() {
  const merge = jsonSchemaToGrammar(MEMORIES_MERGE_SCHEMA);
  Assert.stringContains(
    merge,
    'item ::= "{" space item-component-statements-kv "," space item-new-reasoning-kv "," space item-new-statement-kv "}" space',
    "Required properties are emitted in order with no optional tail"
  );
  Assert.stringContains(
    merge,
    'item-component-statements ::= "[" space string ("," space string)+ "]" space',
    "minItems 2 becomes one item plus at least one more"
  );
  Assert.stringContains(
    merge,
    'item-new-statement ::= "\\"" char{0,100} "\\"" space',
    "maxLength bounds the character repetition"
  );

  const filter = jsonSchemaToGrammar({
    type: "object",
    properties: { kept_memories: { type: "array", items: { type: "string" } } },
    required: ["kept_memories"],
    additionalProperties: false,
  });
  Assert.stringContains(
    filter,
    'root ::= "{" space kept-memories-kv "}" space',
    "A single required array property"
  );
});

add_task(function test_json_grammar() {
  Assert.equal(
    JSON_GRAMMAR,
    jsonSchemaToGrammar({}),
    "Empty schema is any JSON"
  );
  Assert.stringContains(JSON_GRAMMAR, "root ::= object");
  Assert.stringContains(
    JSON_GRAMMAR,
    "value ::= object | array | string | number | boolean | null"
  );
});

add_task(function test_unsupported_schemas_throw() {
  const unsupported = [
    {
      schema: { $ref: "#/definitions/x", definitions: { x: {} } },
      re: /\$ref/,
    },
    { schema: { type: "string", pattern: "^a$" }, re: /pattern/ },
    { schema: { type: "integer", minimum: 0 }, re: /minimum/ },
    { schema: { type: "string", format: "date" }, re: /format/ },
    { schema: { allOf: [{ type: "object" }] }, re: /allOf/ },
    { schema: { type: "array", items: [{ type: "string" }] }, re: /tuple/ },
    { schema: { enum: [] }, re: /enum/ },
    { schema: { type: "custom" }, re: /type/ },
    { schema: "string", re: /Unsupported JSON schema/ },
    {
      schema: {
        type: "object",
        properties: { a: { type: "string" } },
        additionalProperties: { type: "string" },
      },
      re: /additionalProperties/,
    },
  ];
  for (const { schema, re } of unsupported) {
    Assert.throws(
      () => jsonSchemaToGrammar(schema),
      re,
      `Rejects ${JSON.stringify(schema)}`
    );
  }
});
