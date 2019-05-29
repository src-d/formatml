local utils = import 'utils.libjsonnet';
local embedder_features = [
  { name: 'roles', vocabulary: 'voc_roles', dimension: 48 },
  { name: 'internal_type', vocabulary: 'voc_type', dimension: 48 },
  { name: 'length', vocabulary: 'voc_length', dimension: 32 },
];
local encoder_input_dim = utils.sum(utils.pluck(embedder_features, 'dimension'));
local encoder_message_dim = encoder_input_dim;
local encoder_output_dim = encoder_input_dim;
local encoder_edge_types = ['child', 'parent', 'previous_token', 'next_token'];
local encoder_iterations = 8;
local output_embedding_dim = 32;
local decoder_hidden_dim = 128;

utils.class(
  'main',
  {
    run_dir: utils.resource('run_dir'),
    config: {
      dataset: utils.class(
        'repositories',
        {
          local root_dir = 'rootdir',
          download_dir: std.join('/', [root_dir, 'download']),
          parse_dir: std.join('/', [root_dir, 'parse']),
          tensor_dir: std.join('/', [root_dir, 'tensor']),
          repositories: [
            // ['telescopejs', 'telescope', 'master'],
            ['axios', 'axios', 'master'],
          ],
          instance: {
            fields: {
              graph: utils.class(
                'typed_dgl_graph',
                {
                  edge_types: encoder_edge_types,
                  vocabulary: utils.resource('voc_graph'),
                },
              ),
              roles: utils.class('roles', { vocabulary: utils.resource('voc_roles') }),
              internal_type: utils.class(
                'internal_type',
                { vocabulary: utils.resource('voc_type') }
              ),
              length: utils.class(
                'length',
                {
                  max_length: 128,
                  vocabulary: utils.resource('voc_length'),
                },
              ),
              label: utils.class('label', { vocabulary: utils.resource('voc_label') }),
            },
          },
          parser: utils.class('javascript'),
        },
      ),
      model: utils.class(
        'gnn_rnn',
        {
          graph_embedder: {
            dimensions: utils.pluck(embedder_features, 'dimension'),
            vocabularies: std.map(
              utils.resource,
              utils.pluck(embedder_features, 'vocabulary')
            ),
          },
          graph_encoder: utils.class(
            'ggnn',
            {
              iterations: encoder_iterations,
              n_types: std.length(encoder_edge_types),
              x_dim: encoder_input_dim,
              h_dim: encoder_output_dim,
              m_dim: encoder_message_dim,
            },
          ),
          output_embedder: {
            dimension: output_embedding_dim,
            vocabulary: utils.resource('voc_label'),
          },
          decoder: utils.class(
            'concat_conditioning_recurrent',
            {
              recurrent: utils.class(
                'lstm',
                {
                  input_size: encoder_output_dim + output_embedding_dim,
                  hidden_size: decoder_hidden_dim,
                  batch_first: true,
                },
              ),
            }
          ),
          class_projection: {
            in_features: decoder_hidden_dim,
            vocabulary: utils.resource('voc_label'),
          },
          graph_field_name: 'graph',
          feature_field_names: utils.pluck(embedder_features, 'name'),
          label_field_name: 'label',
        },
      ),
      optimizer: utils.class('adam', {}),
      trainer: {
        epochs: 100,
        batch_size: 2,
        eval_every: 20,
        train_eval_split: 0.8,
        run_dir: utils.resource('run_dir'),
        metric_names: ['cross_entropy', 'perplexity', 'accuracy'],
      },
      scheduler: utils.class('step_lr', { step_size: 5, gamma: 0.8 },),
    },
  },
  {
    run_dir: utils.class('date_template_path',
                         { date_template: 'runs/meta-learning/%m-%d-%H:%M:%S%z' }),
    voc_graph: utils.class('vocabulary'),
    voc_roles: utils.class('vocabulary', { unknown: '<UNK>' }),
    voc_type: utils.class('vocabulary', { unknown: '<UNK>' }),
    voc_length: utils.class('vocabulary'),
    voc_label: utils.class('vocabulary', { unknown: '<UNK>' }),
  }
)
