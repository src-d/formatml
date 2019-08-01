import React, { useMemo } from "react";
import "./css/Span.css";

export enum SpanType {
  Error = "error",
  First = "first",
  Predicted = "predicted",
  NotPredicted = "not-predicted"
}

export interface IProps {
  code: string;
  types: SpanType[];
}

const transform = (code: string): (string | JSX.Element)[] => {
  const elements = [];
  const formattings = [];
  const indexes = [];
  const lengths = [];
  const pattern = /[\n\t ]+/g;
  let match = null;
  while ((match = pattern.exec(code)) !== null) {
    formattings.push(match[0]);
    indexes.push(match.index);
    lengths.push(match[0].length);
  }
  let currentIndex = 0;
  for (let i = 0; i < formattings.length; i++) {
    const index = indexes[i];
    const length = lengths[i];
    const formatting = formattings[i];
    if (currentIndex < index) {
      elements.push(code.slice(currentIndex, index));
    }
    elements.push(
      <span key={index} className="formatting">
        {formatting
          .replace(/ /g, "␣")
          .replace(/\n/g, "⏎\n")
          .replace(/\t/g, "⇥")}
      </span>
    );
    currentIndex = index + length;
  }
  if (currentIndex < code.length) {
    elements.push(code.slice(currentIndex));
  }
  return elements;
};

const Span = (props: IProps) => {
  const transformed = useMemo(() => transform(props.code), [props.code]);
  return <span className={props.types.join(" ")}>{transformed}</span>;
};

export default Span;
