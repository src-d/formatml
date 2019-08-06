interface Metadata {
  columns: string[];
  metadata: Record<string, Record<string, string[]>>;
}

interface Data {
  numberOfTasks: number;
  metadata: Metadata;
}

interface Task {
  content: string;
  ranking: number[];
  errorOffset: number;
}
