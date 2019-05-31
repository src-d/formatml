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
local dataset_dir = 'cache-codrep';

{
  context_cache_dir: std.join('/', [dataset_dir, 'resources_cache']),
  command: utils.class(
    'main',
    {
      run_dir: utils.resource('run_dir'),
      config: {
        dataset: utils.class(
          'codrep',
          {
            parse_dir: std.join('/', [dataset_dir, 'parse']),
            tensor_dir: std.join('/', [dataset_dir, 'tensor']),
            input_dir: '/home/mog/work/codrep-2019/Datasets/learning',
            instance: {
              fields: {
                graph: utils.class(
                  'typed_dgl_graph',
                  {
                    edge_types: encoder_edge_types,
                    vocabulary: utils.resource('voc_graph'),
                  },
                ),
                roles: utils.class(
                  'roles',
                  { vocabulary: utils.resource('voc_roles') }
                ),
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
                label: utils.class('binary_labels'),
              },
            },
            parser: utils.class('java', { split_formatting: true }),
          },
        ),
        model: utils.class(
          'gnn_ff',
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
            class_projection: utils.class(
              'linear',
              { in_features: encoder_output_dim, out_features: 2 },
            ),
            graph_field_name: 'graph',
            feature_field_names: utils.pluck(embedder_features, 'name'),
            label_field_name: 'label',
          },
        ),
        optimizer: utils.class('adam', {}),
        trainer: {
          epochs: 100,
          batch_size: 2,
          eval_every: 2000,
          train_eval_split: 0.95,
          run_dir: utils.resource('run_dir'),
          metric_names: ['cross_entropy', 'perplexity', 'mrr'],
        },
        scheduler: utils.class('step_lr', { step_size: 5, gamma: 0.8 },),
      },
    },
    {
      run_dir: utils.class('date_template_path',
                           { date_template: 'runs/codrep/%m-%d-%H:%M:%S%z' }),
      voc_graph: utils.class('vocabulary'),
      voc_roles: utils.class('vocabulary', { unknown: '<UNK>' }),
      voc_type: utils.class('vocabulary', { unknown: '<UNK>' }),
      voc_length: utils.class('vocabulary'),
      voc_label: utils.class('vocabulary', { unknown: '<UNK>' }),
    }
  ),
}
