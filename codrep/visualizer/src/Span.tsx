import React, {
  useMemo,
  useRef,
  useImperativeHandle,
  forwardRef,
  useContext
} from "react";
import "./css/Span.css";
import { ActionTarget, AppContext } from "./App";

export enum SpanType {
  Error = "error",
  First = "first",
  Predicted = "predicted",
  NotPredicted = "not-predicted",
  Selected = "selected"
}

export interface ISpanProps {
  code: string;
  types: SpanType[];
  index?: number;
  selected: boolean;
}

export interface ISpanHandles {
  scrollIntoView(): void;
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

const Span: React.RefForwardingComponent<ISpanHandles, ISpanProps> = (
  props,
  ref
) => {
  const dispatch = useContext(AppContext);
  const transformed = useMemo(() => transform(props.code), [props.code]);
  const spanRef = useRef<HTMLSpanElement>(null);
  useImperativeHandle(ref, () => ({
    scrollIntoView: () => {
      if (spanRef.current !== null) {
        spanRef.current.scrollIntoView();
      }
      if (dispatch !== null) {
        dispatch({
          target: ActionTarget.selectedOffset,
          payload: props.index
        });
      }
    }
  }));
  const attrs =
    props.index !== undefined && dispatch != null
      ? {
          onClick: () =>
            dispatch({
              target: ActionTarget.selectedOffset,
              payload: props.index
            })
        }
      : {};
  return (
    <span
      {...attrs}
      ref={spanRef}
      className={props.types.join(" ") + (props.selected ? " selected" : "")}
    >
      {transformed}
    </span>
  );
};

export default forwardRef(Span);
