interface Data {
  dataset: string;
  tasks: string[];
}

interface Task {
  content: string;
  ranking: number[];
  error_offset: number;
}
