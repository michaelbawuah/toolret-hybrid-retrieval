# Exploratory tool-retrieval failure analysis

This analysis describes the already evaluated 300-query pilot. It does not tune the fixed router or alter binary relevance judgments. It is not confirmation evidence.

## Candidate and ordering effects

| Source | Queries | Hybrid nDCG@10 | CE nDCG@10 | Delta | Wins / losses / same | Candidate label recall@20 | No positive in prefix | Gold left / entered top 10 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| apigen | 100 | 0.611968 | 0.686829 | +0.074861 | 38 / 13 / 49 | 0.8300 | 10 | 5 / 5 |
| toolace | 100 | 0.558829 | 0.555084 | -0.003745 | 19 / 24 / 57 | 0.7442 | 21 | 7 / 4 |
| toolbench | 100 | 0.277550 | 0.253885 | -0.023664 | 26 / 35 / 39 | 0.4770 | 29 | 18 / 12 |

A prefix permutation cannot recover positive tools outside its top-20 candidate set. Candidate misses and reranker ordering errors therefore need different remedies. Candidate label recall is averaged per query and refers to exact labeled tool IDs.

## Token truncation and judgment risks

The exact pinned production tokenizer was replayed at max_length=256, with its longest-first pair truncation. Counts use token_type_ids and the special-token mask; query text and flattened tool text match ranking generation.

| Source | Truncated prefix pairs / total | Truncated positive pairs / positive prefix pairs | Unjudged cross-source top 1 | Possible name-suffix counterpart at top 1 |
|---|---|---|---:|---:|
| apigen | 196 / 2000 | 3 / 107 | 28 | 6 |
| toolace | 559 / 2000 | 15 / 92 | 33 | 3 |
| toolbench | 374 / 2000 | 19 / 104 | 68 | 14 |

A name-suffix counterpart is only a deterministic textual hint: normalized names match or one ends in the other, with the shorter name at least 12 characters. It does not establish functional equivalence, relevance, or label error. The original qrels remain unchanged.

### ToolBench descriptive breakdown

| Outcome | Queries | Mean delta nDCG@10 | Mean query tokens | Mean gold document tokens | Truncated gold prefix pairs / total gold prefix pairs | Unjudged cross-source top 1 |
|---|---:|---:|---:|---:|---|---:|
| loss | 35 | -0.202913 | 52.4 | 129.0 | 5 / 53 | 27 |
| unchanged | 39 | +0.000000 | 58.4 | 113.4 | 1 / 12 | 25 |
| win | 26 | +0.182136 | 58.3 | 184.0 | 13 / 39 | 16 |

Of the 35 ToolBench loss queries, 32 have no truncated positive pair. Among the 14 queries with a truncated positive pair, 10 improve and 3 worsen. This rules out the simple explanation that clipping a labeled positive is necessary for the observed regression; it does not rule out effects from clipped distractors or other interacting factors.

These strata were examined after seeing pilot outcomes. They are descriptive associations, confounded by source, task, annotation, and tool schema; they do not identify causes. Full by-positive-count, query-length, document-length, and candidate-coverage breakdowns are in the JSON artifact.

## Deterministic case review

The five largest ToolBench losses and five largest wins are selected by signed nDCG delta, with query-ID tie breaks. Examples are deliberately outcome-selected and cannot estimate population frequencies.

### toolbench_query_944 (-0.645963 nDCG@10)

The query names Ubidots and asks for two endpoints. CE promotes a general IoT sensor analyzer from another source; the labeled variable-details endpoint moves 1→15. This is a context-versus-endpoint-identity example, not proof of the model's reasoning.

Query excerpt: My company is developing an IoT application and needs to gather insights from the sensor data. Can you help us by providing the variables of a specified data source on Ubidots and their corresponding details, including the last value writte

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_4182 (Ubidots_GET_datasources_datasource_id_variables) | 5 | 7 | 62 | false |
| toolbench_tool_4183 (Ubidots_GET_variables_variable_id) | 1 | 15 | 69 | false |

CE top 1: `toolACE_tool_8821` — IoTDataAnalyzer.analyzeSensorData (hybrid rank 8; labeled positive: False).

### toolbench_query_588 (-0.490363 nDCG@10)

The query requests electricity emissions and fuel prices. CE puts an unjudged carbon-footprint calculator first and moves the labeled fuel-price endpoint 4→18. One request component is retained while the other loses top-10 coverage.

Query excerpt: I'm planning a sustainable wedding and I want to calculate the carbon footprint of our energy usage. Can you provide me with the CO2 emissions per kilowatt-hour for electricity in Germany? Additionally, I would like to know the fuel prices

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_10111 (Electricity_Carbon_Footprint_Germany_CO2_Emission) | 1 | 2 | 127 | false |
| toolbench_tool_8051 (Europe_Fuel_Prices_Get_specific_country) | 4 | 18 | 43 | false |

CE top 1: `toolACE_tool_9352` — EcoDataAnalyzer.calculateCarbonFootprint (hybrid rank 6; labeled positive: False).

### toolbench_query_623 (-0.386853 nDCG@10)

CE places APIGen take_image_screenshot first; the labeled ToolBench Web_Capture_Take_Image_Screenshot moves 2→13. Their visible tool names and descriptions concern the same screenshot operation, but executable equivalence has not been checked. A second gold tool has an empty description and was already absent from fused candidates.

Query excerpt: Please fetch my personal information associated with my account. Can you retrieve my details? Also, I would like to take a screenshot of a specific website, 'https://example.com', with a width of 1200 and height of 800 pixels.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_10505 (Web_Capture_Take_Image_Screenshot) | 2 | 13 | 67 | false |
| toolbench_tool_824 (Demo_Project_v2_Me) | absent | absent | 18 | not scored |

CE top 1: `apigen_tool_1246` — take_image_screenshot (hybrid rank 1; labeled positive: False).

### toolbench_query_366 (-0.347742 nDCG@10)

CE puts an unjudged anime-ranking endpoint first. Both labeled anime-detail/filter endpoints are demoted; one crosses 6→12. The request mixes ranking, current-airing constraints, and detailed metadata. Truncation statistics are recorded separately and do not establish why the score changed.

Query excerpt: I'm planning a movie night with my friends and we want to watch an anime. Can you recommend an anime that has a high ranking, good reviews, and is currently airing? Also, provide us with the synopsis, number of episodes, and the main pictur

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_4667 (animes_Get_anime_detail) | 3 | 10 | 254 | true |
| toolbench_tool_4668 (animes_Get_animes) | 6 | 12 | 653 | true |

CE top 1: `apigen_tool_1183` — get_one_anime_by_ranking (hybrid rank 1; labeled positive: False).

### toolbench_query_497 (-0.344347 nDCG@10)

CE places an unjudged APIGen trivia endpoint first. The labeled trivia endpoint remains at rank 3, while the independent Steam special-offers endpoint moves 4→12. This is a multi-request coverage loss with a possible cross-source counterpart.

Query excerpt: I'm organizing a pub quiz night and I need some challenging trivia questions to keep the participants engaged. Can you provide me with twenty trivia questions from the 'language' and 'geography' categories? Furthermore, I would like to expl

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_3604 (SteamGames_Special_offers_GamesList) | 4 | 12 | 72 | false |
| toolbench_tool_9731 (Trivia_by_API_Ninjas_v1_trivia) | 2 | 3 | 128 | false |

CE top 1: `apigen_tool_91` — v1_trivia (hybrid rank 1; labeled positive: False).

### toolbench_query_154 (+0.646979 nDCG@10)

Both labeled Trinidad-and-Tobago COVID endpoints improve (9→1 and 12→6). This win is useful alongside loss cases because it prevents treating long-document truncation or multi-request queries as universally harmful.

Query excerpt: My company is planning a business trip to Trinidad and Tobago, and we need to assess the Covid-19 situation there. Can you provide us with the most recent Covid-19 statistics for Trinidad and Tobago? Additionally, we also require the Covid-

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_273 (Trinidad_Covid_19_Statistics_getMostRecentDay) | 9 | 1 | 542 | true |
| toolbench_tool_274 (Trinidad_Covid_19_Statistics_getStatsbyDay) | 12 | 6 | 622 | true |

CE top 1: `toolbench_tool_273` — Trinidad_Covid_19_Statistics_getMostRecentDay (hybrid rank 9; labeled positive: True).

### toolbench_query_896 (+0.428571 nDCG@10)

The labeled email-format checker improves 9→1. The distinct email-existence checker remains at rank 88, outside the reranked prefix. Reranking can fix ordering but cannot rescue candidates it never scores.

Query excerpt: I need to ensure that the email 'jenny.smith@yahoo.com' exists. Additionally, I want to validate the format of the email 'john.doe@gmail.com'.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_2868 (Email_Checkup_email_exist) | 88 | 88 | 60 | not scored |
| toolbench_tool_2869 (Email_Checkup_email_format) | 9 | 1 | 45 | false |

CE top 1: `toolbench_tool_2869` — Email_Checkup_email_format (hybrid rank 9; labeled positive: True).

### toolbench_query_113 (+0.375928 nDCG@10)

Four language-specific social-news tools improve into or within the top 10, despite an unjudged French-news tool being ranked first. A wrong top-1 label can coexist with a strong multi-label nDCG improvement.

Query excerpt: I'm working on a research project about social media trends in different countries. Could you fetch the social media news in Russian, Portuguese, Dutch, and Italian? I need the latest news articles to analyze the trends.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_3258 (OneLike_Social_Media_News_in_Russian) | 11 | 6 | 31 | false |
| toolbench_tool_3259 (OneLike_Social_Media_News_in_Portuguese) | 6 | 3 | 31 | false |
| toolbench_tool_3260 (OneLike_Social_Media_News_in_Dutch) | 12 | 7 | 31 | false |
| toolbench_tool_3262 (OneLike_Social_Media_News_in_Italian) | 9 | 4 | 31 | false |

CE top 1: `toolACE_tool_483` — Social Media News in French (hybrid rank 5; labeled positive: False).

### toolbench_query_119 (+0.334231 nDCG@10)

The two labeled SignNow role/field tools improve (5→2 and 15→9), while an unjudged APIGen get_role_ids tool is first. This again separates exact-ID labels from possible cross-source endpoint counterparts.

Query excerpt: I have a document with ID '123abc' and I need to get the role IDs and field IDs. Can you provide me with the details?

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_5029 (SignNow_Get_field_and_field_invite_id) | 15 | 9 | 47 | false |
| toolbench_tool_5030 (SignNow_Get_role_ids) | 5 | 2 | 359 | true |

CE top 1: `apigen_tool_2279` — get_role_ids (hybrid rank 1; labeled positive: False).

### toolbench_query_361 (+0.270066 nDCG@10)

Both labeled campaign-lead tools improve (2→1 and 8→5). This is a straightforward improvement in the ordering of existing prefix candidates, without adding a new candidate.

Query excerpt: I run an e-commerce website and I want to track the leads generated from a specific campaign on my website. Can you provide me with the list of leads and their metadata for the campaign with the ID 'abc123'? Additionally, it would be helpfu

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_3419 (fomoAPI_Get_Campaign_From_URL) | 2 | 1 | 57 | false |
| toolbench_tool_3420 (fomoAPI_Get_Campaign_Lead_From_ID) | 8 | 5 | 46 | false |

CE top 1: `toolbench_tool_3419` — fomoAPI_Get_Campaign_From_URL (hybrid rank 2; labeled positive: True).

## Research implications

1. Evaluate candidate coverage separately from ordering quality; a reranker cannot rescue absent candidates.
2. Report source-level and multi-label results alongside the aggregate. An improvement on one source can conceal losses on another.
3. Preserve exact-ID metrics, but independently adjudicate possible cross-source counterparts before claiming these are executable failures. Any alternate equivalence-aware evaluation needs frozen judging rules and should be reported separately.
4. Test truncation and tool-document serialization as controlled ablations on future frozen data. Their associations here do not justify declaring either the cause of regression.
5. A learned or heuristic router derived from these pilot observations must be evaluated on untouched confirmation queries. This analysis is not evidence that such a router already works.

## Provenance

```json
{
  "cache_sha256": "a8d1bac8cf0b1afb734c43edff2438fc1d486b7536799192c321d70fe5683c81",
  "queries_sha256": "ff727e5ce10933e0332acd1ad18f0556a8c03353325ce7074ccd6fb11857e9bd",
  "corpus_sha256": "ac192e7a6d2b1930f3f61544748ce81741c372a104674a3520f8214b2ac3c09f",
  "manifest_sha256": "6c61a1f36bd54e7078f43c052169b27b38f5c57f8c12ed2a9a4f9dfae227eab1",
  "script_sha256": "d5bbd736aa59c226832f427b34c3d3b2cb0741ac59dd58afeccbb69d8bf11688",
  "text_serialization_sha256": "70bd54a54acf7da4c13d8a5c2e5b379f81b64f19f248681973951c79b99f0750",
  "labels_altered": false,
  "analysis_modifies_rankings_or_router": false
}
```
