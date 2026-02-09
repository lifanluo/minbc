import concurrent.futures
import os
import pickle
import natsort
import numpy as np


def from_pickle(config, path, load_img=True):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # ensure backward/alternate key names are available for downstream code
    # map common alternate names used in different dataset exports to the
    # canonical keys expected elsewhere in the repo
    if "action" not in data and "actions" in data:
        # ensure action length matches config expectation; trim if longer
        try:
            base_dim = config.data.base_action_dim
            a = data["actions"]
            if hasattr(a, "shape") and len(a) >= base_dim:
                data["action"] = np.asarray(a[:base_dim])
            else:
                data["action"] = np.asarray(a)
        except Exception:
            data["action"] = data["actions"]
    # joint positions: many datasets store full `states`; canonical code expects
    # `joint_positions` of length 12. Try to extract a sensible slice when
    # possible (take indices 12:24 if states is longer than 12).
    if "joint_positions" not in data and "states" in data:
        s = data["states"]
        try:
            if len(s) >= 24:
                data["joint_positions"] = np.asarray(s[12:24])
            elif len(s) >= 12:
                data["joint_positions"] = np.asarray(s[:12])
        except Exception:
            pass
    # joint velocities mapping
    if "joint_velocities" not in data and "joint_vel" in data:
        jv = data["joint_vel"]
        try:
            data["joint_velocities"] = np.asarray(jv[:12])
        except Exception:
            pass
    # hand position mapping
    if "xhand_pos" not in data and "hand_states" in data:
        try:
            data["xhand_pos"] = np.asarray(data["hand_states"])
        except Exception:
            pass

    if "base_rgb" not in data and load_img:
        # note: base_rgb is only for HATO compatibility
        rgb_keys = config.data.im_key
        rgb = [data[k] for k in rgb_keys]
        data["base_rgb"] = np.stack(rgb, axis=0)
        for ik in config.data.im_key:
            del data[ik]
    return data


# Get the trajectory data from the given directory
def iterate(path, config, workers=32, load_img=True, num_cam=3):
    dir = os.listdir(path)
    dir = [d for d in dir if d.endswith(".pkl")]
    dir = natsort.natsorted(dir)
    dirname = os.path.basename(path)
    root_path = "./mask_cache"
    data = []

    for i, file in enumerate(dir):
        try:
            # Process each file sequentially
            d = from_pickle(config, os.path.join(path, file), load_img)
            # TODO: ask toru
            # Uncomment the following block if needed
            # if not d["activated"]["l"] and not d["activated"]["r"]:
            #     continue

            basedirfile = os.path.join(dirname, file)
            maskfile = os.path.join(root_path, basedirfile)

            # Check if mask file exists and load it
            if os.path.exists(maskfile):
                d["mask"] = from_pickle(maskfile)

            # Add paths to the data dictionary
            d["mask_path"] = maskfile
            d["file_path"] = os.path.join(path, file)

            # Append the processed data
            data.append(d)

        except Exception as e:
            # Print error message for failed files
            print(f"Failed to load {file}: {e}")
            pass

    # with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    #     futures = {executor.submit(from_pickle, os.path.join(path, file), load_img, num_cam): (i, file) for i, file in enumerate(dir)}
    #     for future in futures:
    #         try:
    #             i, file = futures[future]
    #             d = future.result()
    #             # TODO: ask toru
    #             # if not d["activated"]["l"] and not d["activated"]["r"]:
    #             #     continue
    #             basedirfile = os.path.join(dirname, file)
    #             maskfile = os.path.join(root_path, basedirfile)
    #             if os.path.exists(maskfile):
    #                 d["mask"] = from_pickle(maskfile)
    #             d["mask_path"] = maskfile
    #             d["file_path"] = os.path.join(path, file)
    #             data.append(d)
    #         except:
    #             print(f"Failed to load {file}")
    #             pass
    return data


# Get all trajectory directories from the given path
def get_epi_dir(path, prefix=None):
    dir = natsort.natsorted(os.listdir(path))
    if prefix is not None:
        prefixs = prefix.split("-")

    new_dir = []

    for d in dir:
        if os.path.isdir(os.path.join(path, d)):
            matched = False
            if prefix is None:
                matched = True
            else:
                for prefix in prefixs:
                    if d.startswith(prefix):
                        matched = True
            if matched:
                new_dir.append(d)

    print("All Directories")
    print(new_dir)
    print("==========")
    dir = new_dir
    dir_list = [
        os.path.join(path, d) for d in dir if os.path.isdir(os.path.join(path, d))
    ]
    return dir_list