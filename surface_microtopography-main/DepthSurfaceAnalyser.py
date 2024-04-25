"""
Main file
Author: Mihailo Azhar

If used please cite:
Azhar, Mihailo, et al. "An RGB-D framework for capturing soft‐sediment microtopography." 
Methods in Ecology and Evolution 13.8 (2022): 1730-1745.
"""

import PySimpleGUI as sg
import PIL
import PIL.Image as Image
import os.path
import io
import time
import csv
import ExtractSurfaceStats as ESS
import numpy as np
import base64
import sys

def normalize(arr):
    """
    Linear normalization
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    # Gray or colour
    if len(arr.shape) < 3:
        channel_max = 1
    else:
        channel_max = 3 
        #TODO: properly handle if it's not an image you expect
    for i in range(channel_max):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def get_file_list(folder):
    list_of_files = []
    return list_of_files

def convert_to_bytes(file_or_bytes, resize=None, mode=1):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    '''
    if isinstance(file_or_bytes, str):
        img =Image.open(file_or_bytes)
    else:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = Image.open(dataBytesIO)
    
    # Normalise so depth images appear
    if mode==2:
        img = Image.fromarray(normalize(np.array(img)).astype('uint16'))
    elif mode==3:
        img = Image.fromarray(normalize(np.array(img)).astype('uint32'))
    else:
        img = Image.fromarray(normalize(np.array(img)).astype('uint8'))

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
        cur_width = new_width
        cur_height = new_height
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue(), (cur_width,cur_height)
    
def make_window(theme):
    column_width_max = 40
    coordinate_box_size = (10,1)
    sg.theme(theme)
    menu_def = [['&File', ['!&Load CSV', 'E&xit']],
                ['&Help', ['&About']] ]
    right_click_menu_def = [[], ['Nothing','More Nothing','Exit']]

    file_io_layout = sg.Frame('File Options',[[sg.Menu(menu_def,key = '-MENU-')],
                [sg.Text('Input folder:', size=(8, 1))],
                [sg.Text('Folder'), sg.In(size=(column_width_max-15,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse(key='-CURRENT_FOLDER-')],
                [sg.Text('Images', size=(8, 1))],
                [sg.Listbox(values=[], enable_events=True, size=(column_width_max,20),key='-FILE LIST-')]])
    
    flags = sg.Frame('Parameters',[[sg.CB('Batch Process', size=(12, 1), default=False, key = '-BP-')],                 
                [sg.CB('Save Detrended', size=(12, 1),default=True, key = '-SAVE DT-'),
                sg.CB('Write Results', size=(12, 1), default=True,key = '-WRITE-')]],  element_justification='c')
                #sg.CB('Apply ROI to all', size=(12, 1), enable_events=True, default='0', key='-ROI ALL-')],

    visibility_layout = sg.Frame('Display Mode',[[sg.Slider(orientation ='horizontal', key='-MODE-',size=(10,30),
                 range=(1,3),enable_events=True)]])

    output = sg.Frame('Output',[[sg.Output(size=(column_width_max,20),expand_x=True,expand_y=True, font='Courier 8')]])
    
    operation_layout = sg.Frame('ROI',[[sg.Button('Apply ROI to all', key = '-ROI ALL-'),sg.Button('Clear all ROI',key = '-CLEAR ROI-')]])

    run_layout = sg.Frame('Execute',[[sg.Button('Run', size = (15,2), key='-EXE-')],
                                    [sg.Button('Test', size = (15,2), key='-TEST-')]])

    images_col = [[sg.Text('Choose from the list')],
              [sg.Text(size=(40,1), key='-TOUT-')],
             [sg.Graph(
            canvas_size=(400,400),
            graph_bottom_left=(0, 400),
            graph_top_right=(400, 0),
            key='-IMAGE-',drag_submits=True, enable_events=True,expand_x=True,expand_y=True,right_click_menu=[[],['Erase Selection',]])],
            [sg.Text("Top left: "),sg.Input(size=coordinate_box_size, enable_events=False, key='-START-'),
            sg.Text("Bottom Right: "),sg.Input(size=coordinate_box_size, enable_events=False,  key='-STOP-'),
            sg.Text("Box: "),sg.Input(size=coordinate_box_size, enable_events=False, key='-BOX-')]]

    left = [[file_io_layout],[visibility_layout],[flags],[operation_layout],[run_layout],[output]]
    # Image Layout

    #Excute calculation layout
    # Bring elements together
    layout = [[sg.Column(left,justification='left',vertical_alignment='top'), sg.VSeparator(),sg.Column(images_col)]]
    return sg.Window('Depth Surface Analyser',layout, right_click_menu=right_click_menu_def,resizable=True,auto_size_text=True,auto_size_buttons=True,finalize=True)

def get_rect_coordinates(x0,y0,x1,y1):
    #Find the top left and bottom right coord
    tl_x = min(x0,x1)
    tl_y = min(y0,y1)
    br_x = max(x0,x1)
    br_y = max(y0,y1)
    return (tl_x,tl_y), (br_x,br_y)

def roi_boundary_check(start_point, end_point, imageHeight, imageWidth):
    top_left, btm_right = get_rect_coordinates(start_point[0],start_point[1],end_point[0],end_point[1])
    
    tl_x = 0 if top_left[0] < 0 else top_left[0]
    tl_y = 0 if top_left[1] < 0 else top_left[1]
    br_x = imageWidth-1 if btm_right[0] >= imageWidth else btm_right[0]
    br_y = imageHeight-1 if btm_right[1] >= imageHeight else btm_right[1]
    # Return new top_left,btm_right of rectangle
    return (tl_x, tl_y), (br_x, br_y)

def update_rect_coordinates(x0, y0, x1, y1, window):
    top_left, btm_right = get_rect_coordinates(x0,y0,x1,y1)
    window['-START-'].update(f'{top_left[0]},{top_left[1]}')
    window['-STOP-'].update(f'{btm_right[0]},{btm_right[1]}')
    window['-BOX-'].update(f'{abs(btm_right[0]-top_left[0]+1)}, {abs(btm_right[1]-top_left[1]+1)}')

def redraw_rectangle(element, rect_id, start, stop, colour):
    if rect_id is not None:
        element.delete_figure(rect_id)
    return element.draw_rectangle(start,stop, line_color=colour)

def save_roi_csv(ROI_dict, root_folder, time_str):
    csv_filename = root_folder + '/surface_rois_'+ time_str + '.csv'

    with open(csv_filename, 'w',newline='') as csv_file_in:
        writer = csv.writer(csv_file_in)
        writer.writerow(['Root folder','Filename','Top Left','Bottom Right'])
        for key, values in ROI_dict.items():
            writer.writerow([root_folder,key,values[0],values[1]])
        csv_file_in.close()

def save_surface_stats_csv(headers, stats, full_filename):
    with open(full_filename, 'w',newline='') as csv_file_in:
        writer = csv.writer(csv_file_in)
        writer.writerow(headers)
        writer.writerows(stats)
        csv_file_in.close()

def load_roi_csv(csv_file,im_roi_dict):
    if csv_file is not None and csv_file != '':
        with open(csv_file, mode='r') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            if len(header) > 4:
                sg.Popup('Oops!', 'This does not look like the expected CSV file.\nPlease load an ROI csv file.')
                return
            for rows in reader:
                #redraw_rectangle(element, rect_id, start, stop, colour)
                im_roi_dict[rows[1]] = [eval(rows[2]), eval(rows[3]), None]

def enable_load_csv(window):
    menu_def = window['-MENU-'].MenuDefinition
    menu_def[0][1][0] = '&Load CSV'
    window['-MENU-'].update(menu_definition=menu_def)

def process_surface(current_folder, processing_dict):
    stats_dict = {}
    i = 0
    sg.one_line_progress_meter(f'Processing', i, len(processing_dict), f'Processing surface')
    
    bulk_stats= np.empty((0,17), np.float64) # BULK PROCESSING  
    
    for im_file, values in processing_dict.items():  
        current_filename = current_folder + '/' + im_file
        roi =  [values[0], values[1]]
        surface_im = Image.open(current_filename).crop((roi[0][0],roi[0][1],roi[1][0],roi[1][1]))
        if surface_im.mode == 'RGB' or surface_im.mode == 'RGBA':
            print(f'WARN: {im_file} is not a depth image.\nSkipping processing.')
            sg.one_line_progress_meter(f'Processing', i, len(processing_dict), f'Processing surface') 
            i += 1
            continue
        stats, stats_num, normalised_detrended = ESS.extract_surface_stats(surface_im)
        ########### ALERT CHECK IF IT READS 16bit

        stats_dict[im_file] = (current_folder + '/', im_file , *stats)
        print('******************')
        print(im_file)
        print(f'Plot Mean:{stats[0]}')
        print(f'Positive amplitude mean:{stats[1]}')
        print(f'Positive amplitude std:{stats[2]}')
        print(f'Negative amplitude mean:{stats[3]}')
        print(f'Negative amplitude std:{stats[4]}')
        print(f'Arithmetical Mean Deviation:{stats[5]}')
        print(f'RMS deviation:{stats[6]}')
        print(f'Skew:{stats[7]}')
        print(f'Kurtosis:{stats[8]}')
        print(f'Sk:{stats[9]}')
        print(f'Spk:{stats[10]}')
        print(f'Svk:{stats[11]}')
        print(f'Spk/Sk:{stats[12]}')
        print(f'Spk/Sk:{stats[13]}')
        print(f'Angle start:{stats[14]}')
        #print(f'Angle 2:{stats[15]}')
        print(f'Angle final:{stats[16]}')
        print('******************')
        i += 1
        sg.one_line_progress_meter(f'Processing', i, len(processing_dict), f'Processing surface')   
        detrended_filename = current_folder + '/detrended_' + im_file
        result = Image.fromarray(normalised_detrended.astype(np.uint8))
        result.save(detrended_filename)
        bulk_stats = np.append(bulk_stats, [stats_num], axis=0)
    bulk_mean = np.mean(bulk_stats, axis=0)
    stats_dict["Mean"] = (' ', 'Mean', *bulk_mean)
    return stats_dict

def test_functionality(current_filename, roi):
    surface_im = Image.open(current_filename).crop((roi[0][0],roi[0][1],roi[1][0],roi[1][1]))
    if surface_im.mode == 'RGB' or surface_im.mode == 'RGBA':
        print(f'WARN: {im_file} is not a depth image.\nSkipping processing.')
        #sg.one_line_progress_meter(f'Processing', i+1, len(processing_dict), f'Processing surface') 
    test_out = ESS.get_abbott_stats(surface_im, np.size(surface_im)[1],np.size(surface_im)[0])


def Run():
    
    # TODO: Read ini file for last settings

    # TODO: Tighten the layout (do this last)
    #layout = [[sg.Graph((600,450),(0,450), (600,0), key='-GRAPH-', enable_events=True, drag_submits=True)],],

    ## UI Globals
    #cur_image = None
    window = make_window(sg.theme('DarkGrey4'))
    window.bind('<Configure>',"Event")
    window['-START-'].bind("<Return>", "_Enter")
    window['-STOP-'].bind("<Return>", "_Enter")
    window['-BOX-'].bind("<Return>", "_Enter")
    #window.Finalize()
    graph_view = window['-IMAGE-']
    
    ## view-model globals
    dragging = False
    start_point = (0,0)
    end_point = (0,0)
    prior_rect = None
    im_roi_dict = {}
    current_surf_filename = ''
    fnames = []
    imageHeight = 0
    imageWidth = 0
    while True:
        event, values = window.read(timeout=100)
        #print(event)
        
        ## Event handling decision making
        # Exit app
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        elif event == 'About':
            sg.popup('Depth surface analyser',
                     'Institute of Marine Science',
                     'Intelligent Vision Systems Laboratory',
                     'Author: Mihailo Azhar',
                     'Version: 1.0.0',
                     'Date: 22/09/22')

        elif event == 'Load CSV':
            csv_file = sg.PopupGetFile('Select csv', default_path = values['-FOLDER-'], file_types=(("CSV Files","*roi*.csv"),),no_window=True)
            load_roi_csv(csv_file,im_roi_dict)
            # Draw (make a function?)
            if  current_surf_filename in im_roi_dict:
                    start_point, end_point, prior_rect = im_roi_dict[current_surf_filename]
                    prior_rect = redraw_rectangle(graph_view, prior_rect, start_point, end_point, 'green2')

        ## Folder name was filled in, make a  list of files in the folder
        if event == '-FOLDER-':                         
            folder = values['-FOLDER-']

            try:
                # get list of files in folder
                file_list = os.listdir(folder)         
            except:
                file_list = []
            finally:
                #Changing folders clears the ROI list
                im_roi_dict = {}
                start_point = (0,0)
                end_point = (0,0)
                prior_rect = None

            fnames = [f for f in file_list if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", "jpeg", ".tiff", ".bmp"))]
            window['-FILE LIST-'].update(fnames)
            window['-CURRENT_FOLDER-'].InitialFolder = folder
            # enable load csv
            enable_load_csv(window)

        elif event in ('-FILE LIST-','-MODE-'):    # A file was chosen from the listbox
            try:
                current_surf_filename = values['-FILE LIST-'][0]
                surf_filename_full = os.path.join(values['-FOLDER-'], current_surf_filename)
                window['-TOUT-'].update(surf_filename_full)

                # Clear graph because we are swapping views
                graph_view.erase()

                ## Draw
                # Grab new image
                image_data,im_size = convert_to_bytes(surf_filename_full,resize=None,mode=values['-MODE-'])
                graph_view.set_size(im_size)
                graph_view.change_coordinates((0,im_size[1]-1),(im_size[0]-1,0))
                graph_view.DrawImage(data=image_data,location=(0,0))
                window['-CURRENT_FOLDER-'].InitialFolder = folder

                # Update current image globals
                start_point = (0,0)
                end_point = (0,0)
                prior_rect = None
                imageWidth = im_size[0]
                imageHeight = im_size[1]
                
                # Check the dict for a rectangle
                if  current_surf_filename in im_roi_dict:
                    start_point, end_point, prior_rect = im_roi_dict[current_surf_filename]
                    prior_rect = redraw_rectangle(graph_view, prior_rect, start_point, end_point, 'green2')
            
            except Exception as E:
                print(f'** Error {E} **')
                pass        # something weird happened making the full filename

        if event in ('-IMAGE-', '-IMAGE-+UP'):
            x, y = values['-IMAGE-']
            if not dragging:
                # First mouse down even on graph element
                start_point = (x,y)
                dragging = True
                # Check dictionary if we have drawn something before. Clear it
                if prior_rect is not None:
                    graph_view.delete_figure(prior_rect)
                
            elif dragging:
                # Update the ROI
                prior_rect = redraw_rectangle(graph_view, prior_rect, start_point, (x,y), 'DeepSkyBlue')

            update_rect_coordinates(start_point[0], start_point[1], x, y, window)
            
            if event.endswith('+UP'):
                x, y = values['-IMAGE-']
                end_point = (x,y)
                
                # Check points are within boundaries
                start_point, end_point = roi_boundary_check(start_point, end_point, imageHeight, imageWidth)
                update_rect_coordinates(start_point[0], start_point[1], end_point[0], end_point[1], window)

                dragging = False
                #prior_rect = redraw_rectangle(graph_view, prior_rect, start_point, (x,y), 'green2')
                prior_rect = redraw_rectangle(graph_view, prior_rect, start_point, end_point, 'green2')

                # Update the roi dict
                #im_roi_dict[current_surf_filename] = [start_point, (x,y), prior_rect]
                im_roi_dict[current_surf_filename] = [start_point, end_point, prior_rect]

        if event in ('-START-_Enter', '-STOP-_Enter', '-BOX-_Enter'):
            # Grab text inputs
            try:
                start_point = tuple(map(int,window['-START-'].get().strip().split(',')))
                end_point = tuple(map(int,window['-STOP-'].get().strip().split(',')))
                box = tuple(map(int,window['-BOX-'].get().strip().split(',')))

                # Adjust based on changed values
                if event == '-START-_Enter' or event == '-STOP-_Enter':
                    window['-BOX-'].update(f'{abs(end_point[0]-start_point[0]+1)}, {abs(end_point[1]-start_point[1]+1)}')
                elif event == '-BOX-_Enter':
                    window['-STOP-'].update(f'{abs(end_point[0]-start_point[0]+1)}, {abs(end_point[1]-start_point[1]+1)}')
                    end_point = (start_point[0] + box[0], start_point[1] + box[1])
                prior_rect = redraw_rectangle(graph_view, prior_rect, start_point, end_point, 'green2')

                # Update the roi dict
                im_roi_dict[current_surf_filename] = [start_point, end_point, prior_rect]

            except ValueError:
                sg.Popup('Oops!', 'Please use integer values')

        if event == '-ROI ALL-':
            if len(fnames) > 0 and prior_rect is not None:
                im_roi_dict = {k: [start_point, end_point, prior_rect] for k in fnames}
        elif event == '-CLEAR ROI-':
            im_roi_dict = {}
            if prior_rect is not None:
                graph_view.delete_figure(prior_rect)
        elif event == '-EXE-':
            if len(im_roi_dict) > 0:
                # Grab time
                time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
                # Check the inputs
                if values['-BP-'] == True:
                    # Batch Process
                    #TODO ask if they want to apply ROI to all if 
                    stats_dict = process_surface(values['-FOLDER-'], im_roi_dict)
                else:
                    # Single Process
                    if prior_rect is not None:
                        stats_dict = process_surface(values['-FOLDER-'], {current_surf_filename:[start_point, end_point, prior_rect]})
                if values['-WRITE-'] == True:
                    headers = ('Folder', 'File', 'Plot Mean','Positive topography mean','Positive topography sd','Negative topography mean', 'Negative topography SD', 'AMD','RMS deviation','Skew',
                    'Kurtosis','Sk','Spk','Svk','Spk/Sk','Svk/Sk','Angle 1','Angle 2','Final Angle')
                    save_surface_stats_csv(headers, stats_dict.values(), values['-FOLDER-'] + '/surface_stats_' + time_str + '.csv')
                    save_roi_csv(im_roi_dict, values['-FOLDER-'], time_str)
                print ('Processing complete!')            
            #TODO: Make saving a standard method (get rid of save_roi_csv)
            #headers = ('Root folder','Filename','Top Left','Bottom Right','Id')
            #save_surface_stats_csv(headers, im_roi_dict.values(), values['-FOLDER-'] + '/saved_surface_rois_' + time_str + '.csv')
        elif event =='-TEST-':
            test_functionality(values['-FOLDER-'] + '/' + current_surf_filename, im_roi_dict[current_surf_filename])

    window.close()


if __name__ == "__main__":
    print("Starting depth surface analyser")
    Run()
print("Closing application")
sys.exit(0)

