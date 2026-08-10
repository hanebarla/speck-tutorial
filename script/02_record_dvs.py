import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import samna
import sinabs.backend.dynapcnn.io as sio


DEVICE = "speck2fdevkit:0"


def dvs_events_to_numpy(events: List[samna.speck2f.event.DvsEvent]) -> np.ndarray:
    dtype = np.dtype([("x", "u1"), ("y", "u1"), ("p", bool), ("t", "u4")])
    return np.array(
        [
            (ev.x, ev.y, ev.p, ev.timestamp)
            for ev in events
            if isinstance(ev, samna.speck2f.event.DvsEvent)
        ],
        dtype=dtype,
    )


def main():
    devkit = sio.open_device(DEVICE)

    # devkitから出てくるイベントを読むためのsinkを作る
    sink = samna.graph.sink_from(devkit.get_model_source_node())

    # DVSのraw eventをモニタできるように設定する
    config = samna.speck2f.configuration.SpeckConfiguration()
    config.dvs_layer.raw_monitor_enable = True
    devkit.get_model().apply_configuration(config)

    # timestampを有効化
    stopwatch = devkit.get_stop_watch()
    stopwatch.start()
    stopwatch.reset()

    print("DVS eventsを1秒間取得します。Speckの前で手を動かしてください。")
    sink.clear_events()
    time.sleep(1.0)

    events = dvs_events_to_numpy(sink.get_events())
    print(f"取得イベント数: {len(events)}")

    frame = np.zeros((128, 128), dtype=np.float32)
    if len(events) > 0:
        np.add.at(frame, (events["x"], events["y"]), 1.0)

    plt.figure()
    plt.imshow(frame.T)
    plt.title("Accumulated DVS events for 1 s")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("output/dvs_events.png")


if __name__ == "__main__":
    main()