import xml.etree.ElementTree as ET
import glob
import os
import argparse
def update_annotation_labels(xml_file,old_label,new_label):
    #     # Given list with keys
    # key_list = ["Unkown35", "Unknown36", "Unknown37"]

    # # Value to be associated with each key
    # value = 'Waqar'

    # # Create the dictionary using dictionary comprehension
    # my_dict = {key: value for key in key_list}

    # print(my_dict)
    old_label = old_label.split(',')
    # print(type(old_label))
    label_dict = {key: new_label for key in old_label}
    print(label_dict)
    xml_files = glob.glob(os.path.join(xml_file, '*.xml'))
    # Load the XML file
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Update annotation labels
        for annotation in root.iter('object'):
            old_label = annotation.find('name').text
            if old_label in label_dict:
                new_label = label_dict[old_label]
                annotation.find('name').text = new_label

        # Save the updated XML file
        tree.write(xml_file)

# # Example usage
# xml_file_path = 'dataset_generate'
# labels = {
#     'Unknown0': 'Ahsan1',
#     'Unknown1': 'Bilawal1',
# }

# update_annotation_labels(xml_file_path, labels)



if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Data Generation with XML')
    
    # Add argument for video path
    parser.add_argument('xml_path',type=str, help='Path to the video file')
    parser.add_argument('old_names_array', help='Path to the video file')
    parser.add_argument('new_label',type=str, help='Path to the video file')
    # Add argument for frame_number
    # parser.add_argument('frame_number',type=int,default=0,help='frame_number')

    
    # Parse the command-line argumentss
    args = parser.parse_args()
    
    # Call the data_generation_with_xml function with the provided video path
    update_annotation_labels(args.xml_path, args.old_names_array, args.new_label)
    