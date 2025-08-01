


class MergedHighlightsImshow:
    ''''''
    def __init__(self, fig, ax, cell_centers, cell_data, cell_width, cell_height, bbwidth, wbwidth, cmap, vmin, vmax):
        self.cell_centers = cell_centers
        self.cell_data = cell_data
        self.ax = ax
        self.fig=fig
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.bbwidth=bbwidth
        self.wbwidth=wbwidth
        self.cmap = cmap
        self.vmin=vmin
        self.vmax=vmax
        self.norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)


        cell_centers_x, cell_centers_y = cell_centers

        #default values of left, bottom, width, height for each cell - this will be updated when highlights are added, and then merged along x-axis or y-axis
        self.cell_centers_x_arr = cell_centers_x[:,None].repeat(cell_data.shape[1],axis=1)
        self.cell_centers_y_arr = cell_centers_y[None,:].repeat(cell_data.shape[0],axis=0)
        self.cell_lefts         = (cell_centers_x -  cell_width/2)[:,None].repeat(cell_data.shape[1],axis=1)
        self.cell_bottoms       = (cell_centers_y - cell_height/2)[None,:].repeat(cell_data.shape[0],axis=0)
        self.cell_widths        = np.full(cell_data.shape, cell_width,dtype="f")
        self.cell_heights       = np.full(cell_data.shape, cell_height, dtype="f")

        #bbox for the heatmap only (excluding added heatmap title)
        self.heatmap_bbox = [cell_centers_x[0]-cell_width/2,
                     cell_centers_y[0]-cell_height/2,
                     np.sum(self.cell_widths,axis=0)[0],
                     np.sum(self.cell_heights,axis=1)[0]]

        #bbox for heatmap + title (updated later in 'add_title' if title is added)
        self.bbox = [cell_centers_x[0]-cell_width/2,
                     cell_centers_y[0]-cell_height/2,
                     np.sum(self.cell_widths,axis=0)[0],
                     np.sum(self.cell_heights,axis=1)[0]] #bounding box for whole heatmap (left, bottom, width, height)


        #from itertools import product
        #assert all([i==j for i,j in product(*[[self.cell_data.shape, self.cell_left_extents.shape, self.cell_right_extents.shape, self.cell_widths.shape, self.cell_heights.shape]]*2) ])


    def get_text_patch(self, x, y, text, text_height=None, fontsize=None, weight=None, alignment='', center_x=True, center_y=True, rotation=0):

        fp = FontProperties(size=1,weight=weight)
        sf = text_height/bb.height if not text_height is None else fontsize
        tp = TextPath(  (0,0), text, prop=fp).transformed(Affine2D().scale(sf))
        bb = tp.get_extents()
        # width,height=bb.width,bb.height
        # x = (2*self.bbox[0]+self.bbox[2])/2 #where to put center of text (horizonally)
        # y = self.bbox[1]+self.bbox[3] - bb.ymin + gap #where to put bottom of text; we subtract bb.ymin in case the text has underhanging characters (which go below (0,0)), and add the gap beneath (between heatmap and heatmap title)
        offset_x = x-(bb.width/2) #if center_x else x
        offset_y = y-(bb.height/2) #if center_y else y

        # translate to be centered on offset_x,offset_y, and rotate
        tp = tp.transformed(Affine2D().translate(offset_x,offset_y)).transformed(Affine2D().rotate_deg(rotation))#)_around(offset_x,offset_y,rotation))
        bb = tp.get_extents()
        adj_x = bb.width/2 if 'r' in alignment else 0
        adj_x = -bb.width/2 if 'l' in alignment else adj_x #left takes precidence over right since this line is last
        adj_y = bb.height/2 if 'b' in alignment else 0
        adj_y = -bb.height/2 if 't' in alignment else adj_y
        # print(adj_x,adj_y)
        tp = tp.transformed(Affine2D().translate(adj_x,adj_y))

        pt = PathPatch(tp, color="black")
        return tp,pt



    def add_title(self, title, horizontalalignment='center', verticalalignment='bottom', fontsize=1,weight="bold"):
        # ttl = self.ax.text( (2*self.bbox[0]+self.bbox[2])/2, self.bbox[1]+self.bbox[3], title, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, fontsize=fontsize,weight=weight )
        # fontsize=1
        gap = 0.2

        fp = FontProperties(size=fontsize,weight=weight)
        tp = TextPath(  (0,0), title, prop=fp).transformed(Affine2D().scale(fontsize))
        bb = tp.get_extents()
        x = (2*self.bbox[0]+self.bbox[2])/2 #where to put center of text (horizonally)
        y = self.bbox[1]+self.bbox[3] + gap #where to put bottom of text; we subtract bb.ymin in case the text has underhanging characters (which go below (0,0)), and add the gap beneath (between heatmap and heatmap title)
        offset_x = x-(bb.width/2)
        offset_y = y #-(bb.height/2)
        transform = Affine2D().translate(offset_x,offset_y)
        tp = tp.transformed(transform)
        pt = PathPatch(tp, color="black")
        self.ax.add_patch(pt)
        bb = tp.get_extents()
        # print(bb)
        self.bbox[3] = bb.ymax-self.bbox[1] + gap #adjust height of bbox of current heatmap and add the gap above




        # tp = TextPath((  (2*self.bbox[0]+self.bbox[2])/2, self.bbox[1]+self.bbox[3]  ), title, prop=fp)
        # self.ax.add_patch(PathPatch(tp, color="black"))
        # self.bbox[3] += 1

        # self.fig.canvas.draw()
        # xmin = ttl.get_window_extent().xmin
        # xmax = ttl.get_window_extent().xmax
        # ymin = ttl.get_window_extent().ymin
        # ymax = ttl.get_window_extent().ymax
        # xmin, ymin = fig.transFigure.inverted().transform((xmin, ymin))
        # xmax, ymax = fig.transFigure.inverted().transform((xmax, ymax))
        #
        # # self.bbox[3] = self.bbox[3] + fig.transFigure.inverted().transform( ((ttl.get_window_extent().xmin+ttl.get_window_extent().xmax)/2 , ttl.get_window_extent().ymax) )[0]
        # # self.bbox[3] = self.bbox[3] + self.fig.transFigure.inverted().transform(  ttl.get_window_extent().ymax )[1]
        # self.bbox[3] = self.bbox[3] + ymax-ymin


    def merge_boundary(self,xi,yi, direction = "left"):
        # print(xi, yi, "-----------------")
        # print(self.cell_lefts[xi,yi],self.cell_bottoms[xi,yi],self.cell_widths[xi,yi],self.cell_heights[xi,yi])
        if direction == "left":
            new_left = self.cell_centers_x_arr[xi,yi] - self.cell_width/2
            left_shift = self.cell_lefts[xi,yi] - new_left  #actual left minus original left (when the highligher obj was initialized)
            corrected_width = self.cell_widths[xi,yi] + left_shift
            self.cell_lefts[xi,yi],self.cell_widths[xi,yi] = new_left, corrected_width

        elif direction == "bottom":
            new_bottom = self.cell_centers_y_arr[xi,yi] - self.cell_height/2
            bottom_shift = self.cell_bottoms[xi,yi] - new_bottom   #actual bottom minus original bottom (when the highligher obj was initialized)
            corrected_height = self.cell_heights[xi,yi] + bottom_shift
            self.cell_bottoms[xi,yi],self.cell_heights[xi,yi] = new_bottom, corrected_height

        elif direction == "right":
            current_left = self.cell_lefts[xi,yi]
            new_right = self.cell_centers_x_arr[xi,yi] + self.cell_width/2
            new_width = new_right-current_left
            self.cell_widths[xi,yi] = new_width

        elif direction == "top":
            current_bottom = self.cell_bottoms[xi,yi]
            new_top = self.cell_centers_y_arr[xi,yi] + self.cell_height/2
            new_height = new_top-current_bottom
            self.cell_heights[xi,yi] = new_height

        else:
            raise Exception('direction must be chosen from {"left", "bottom", "right", "top"}')
        # print(self.cell_lefts[xi,yi],self.cell_bottoms[xi,yi],self.cell_widths[xi,yi],self.cell_heights[xi,yi])

    def make_merges(self,mask, merge_along_x_axis=True, merge_along_y_axis=False):
        #row indices of mask correspond to x-axis increments   - as you go down the rows, you increment up the units of the x-axis
        #col indices of mask correspond to y-axis increments   - as you go right across cols, you increment up the units of the y-axis
        for xi in range(mask.shape[0]):
            for yi in range(mask.shape[1]):
                # print("------", self.cell_width, self.cell_height)
                if not mask[xi,yi]:
                    continue

                #left   = self.cell_lefts[xi,yi] + highlight_inset
                #bottom = self.cell_bottoms[xi,yi] + highlight_inset
                #width  = self.cell_widths[xi,yi]  - 2*highlight_inset
                #height = self.cell_heights[xi,yi] - 2*highlight_inset


                if merge_along_x_axis:
                    if xi<mask.shape[0]-1:
                        if mask[xi+1,yi]:
                            self.merge_boundary(xi,yi, direction = "right")
                    if xi>0:
                        if mask[xi-1,yi]:
                            self.merge_boundary(xi,yi, direction = "left")

                if merge_along_y_axis:
                    if yi<mask.shape[1]-1:
                        if mask[xi,yi+1]:
                            self.merge_boundary(xi,yi, direction = "top")

                    if yi>0:
                        if mask[xi,yi-1]:
                            self.merge_boundary(xi,yi, direction = "bottom")





                #left   = self.cell_centers_x_arr[xi,yi] - self.cell_width/2  + highlight_inset
                #bottom = self.cell_centers_y_arr[xi,yi] - self.cell_height/2 + highlight_inset
                #width  = self.cell_widths[xi,yi]  - 2*highlight_inset
                #print("111111", width)
                #height = self.cell_heights[xi,yi] - 2*highlight_inset
                #left   = self.cell_lefts[xi,yi] + highlight_inset
                #bottom = self.cell_bottoms[xi,yi] + highlight_inset
                #width  = self.cell_widths[xi,yi]  - 2*highlight_inset
                #height = self.cell_heights[xi,yi] - 2*highlight_inset

                #if merge_along_x_axis and xi>0 and xi<mask.shape[0]-1:
                #    if mask[xi-1,yi]:
                #        left = self.cell_centers_x_arr[xi,yi]-self.cell_width/2
                #    if mask[xi+1,yi]:
                #        width = self.cell_width
                #        print("222222222222",width)

                #if merge_along_y_axis and yi>0 and yi<mask.shape[1]-1:
                #    if mask[xi,yi-1]:
                #        bottom = self.cell_centers_x_arr[xi,yi]-self.cell_height/2
                #    if mask[xi,yi+1]:
                #        height = self.cell_height



                ##print(f"Before assignment: {left}, {bottom}, {width}, {height}")
                #self.cell_lefts[xi, yi] = left
                #self.cell_bottoms[xi, yi] = bottom
                #print(self.cell_widths)
                #self.cell_widths[xi, yi] = width
                #print(self.cell_widths)
                #self.cell_heights[xi, yi] = height
                #print(f"After assignment: {self.cell_lefts[xi, yi]}, {self.cell_bottoms[xi, yi]}, {self.cell_widths[xi, yi]}, {self.cell_heights[xi, yi]}")
                #print(f"Equality check: {self.cell_lefts[xi, yi] == left}, {self.cell_bottoms[xi, yi] == bottom}, {self.cell_widths[xi, yi] == width}, {self.cell_heights[xi, yi] == height}")



                #self.cell_lefts[xi,yi]   = left
                #self.cell_bottoms[xi,yi] = bottom
                #self.cell_widths[xi,yi]  = width
                #self.cell_heights[xi,yi] = height
                #print("555555555555555")
                #print((left,bottom,width,height))

                #print(self.cell_lefts[xi,yi], self.cell_bottoms[xi,yi], self.cell_widths[xi,yi], self.cell_heights[xi,yi])
                #print("666666666666666666666")

    def add_boarder(self,boarder_width=0, clr = 'black', mask=None):
        if not mask is None:
            self.add_mask_boarder(mask, boarder_width=boarder_width,clr=clr)
        else:
            left,bottom,width,height = self.bbox
            left,bottom,width,height = left-boarder_width , bottom-boarder_width , width+boarder_width*2 , height+boarder_width*2
            self.ax.add_patch(Rectangle( (left, bottom) ,  width , height ,color=clr,fill=True, lw=0))
            self.bbox = [left,bottom,width,height]

    def add_mask_boarder(self,mask, boarder_width=0, clr='black'):
        minleft=self.bbox[0]
        minbottom=self.bbox[1]
        maxright=minleft+self.bbox[2]
        maxtop=minbottom+self.bbox[3]
        for xi in range(mask.shape[0]):
            for yi in range(mask.shape[1]):
                if not mask[xi,yi]:
                    continue
                left = self.cell_lefts[xi,yi] - boarder_width
                bottom = self.cell_bottoms[xi,yi] - boarder_width
                width  = self.cell_widths[xi,yi]  + 2*boarder_width
                height = self.cell_heights[xi,yi] + 2*boarder_width
                self.ax.add_patch(Rectangle( (left, bottom) ,  width , height ,color=clr,fill=True, lw=0))


                right, top = left+width, bottom+height
                minleft = min(minleft,left)
                minbottom = min(minbottom,bottom)
                maxright=max(maxright,right)
                maxtop=max(maxtop,top)

        self.bbox = [minleft, minbottom, maxright-minleft, maxtop-minbottom]


    def add_insets(self,mask, highlight_inset=0):
        for xi in range(mask.shape[0]):
            for yi in range(mask.shape[1]):
                if not mask[xi,yi]:
                    continue
                self.cell_lefts[xi,yi] = self.cell_lefts[xi,yi] + highlight_inset
                self.cell_bottoms[xi,yi] = self.cell_bottoms[xi,yi] + highlight_inset
                self.cell_widths[xi,yi]  = self.cell_widths[xi,yi]  - 2*highlight_inset
                self.cell_heights[xi,yi] = self.cell_heights[xi,yi] - 2*highlight_inset

    # def add_highlight(self, mask, clr_as_cell_data=False, clr="black", cmap=None, norm=None, add_text=False,tcolor='white'):
    def add_highlight(self, clr_as_cell_data=False, clr="black", cmap=None, norm=None, add_text=False,tcolor='white', mask=None):
        if clr_as_cell_data and (isinstance(cmap,type(None)) or isinstance(norm,type(None))):
            print(clr_as_cell_data)
            print(type(cmap))
            print(type(norm))
            raise Exception("if clr_as_cell_data is True, a cmap and norm must be passed as keyword args")



        for xi in range(self.cell_data.shape[0]):
            for yi in range(self.cell_data.shape[1]):

                if not mask is None:
                    if not mask[xi,yi]:
                        continue

                left   = self.cell_lefts[xi,yi]
                bottom = self.cell_bottoms[xi,yi]
                width  = self.cell_widths[xi,yi]
                height = self.cell_heights[xi,yi]
                if clr_as_cell_data:
                    clr = self.cmap(self.norm( self.cell_data[xi,yi] ))
                self.ax.add_patch(Rectangle( (left, bottom) ,  width , height ,color=clr,fill=True, lw=0))
                # print((left,bottom,width,height), self.cell_data[xi,yi])
                if add_text:
                    ax.text(self.cell_centers_x_arr[xi,yi],self.cell_centers_y_arr[xi,yi],
                            round(self.cell_data[xi,yi],2), color=tcolor,fontsize=10,horizontalalignment='center',verticalalignment='center')

    def add_all_highlights(self, mask, bbwidth, wbwidth, merge_along_x_axis=True, merge_along_y_axis = False, add_text=False, tcolor ='white'):
        cmap=self.cmap
        norm=self.norm
        # print("---------------------------")
        # print(mask)
        self.add_boarder(boarder_width=self.bbwidth,clr="black")
        self.add_highlight(clr="black")

        self.add_insets(mask,highlight_inset = self.bbwidth)
        self.make_merges(mask, merge_along_x_axis=True, merge_along_y_axis=False)
        self.add_highlight(clr="white")

        self.add_insets(mask,highlight_inset = self.wbwidth)
        self.make_merges(mask, merge_along_x_axis=True, merge_along_y_axis=False)
        self.add_highlight(clr_as_cell_data = True, cmap=self.cmap, norm=self.norm, add_text=add_text, tcolor=tcolor)






    def make_lower_triangle_heatmap(self, group_mask, boarder_mask, bbwidth, wbwidth, merge_along_x_axis=True, merge_along_y_axis = False, add_text=False, tcolor ='white'):
        cmap=self.cmap
        norm=self.norm

        self.add_boarder(boarder_width=self.bbwidth,clr="black", mask=boarder_mask)
        # self.add_highlight(clr="black")
        #
        # self.add_insets(group_mask, highlight_inset = self.bbwidth)
        # self.make_merges(group_mask, merge_along_x_axis=True, merge_along_y_axis=False)
        # self.add_highlight(clr="white")
        #
        # self.add_insets(group_mask,highlight_inset = self.wbwidth)
        # self.make_merges(group_mask, merge_along_x_axis=True, merge_along_y_axis=False)
        self.add_highlight(clr_as_cell_data = True, cmap=self.cmap, norm=self.norm, add_text=add_text, tcolor=tcolor, mask=boarder_mask)






    def make_basic_heatmap(self,add_text=False, tcolor='white'):
        self.add_highlight(clr_as_cell_data = True, cmap=self.cmap, norm=self.norm, add_text=add_text, tcolor=tcolor)


    def _add_signs(self,sign,x,y,size, clr="black"): #sign accepts "+","-"
        if sign=="+":
            self.ax.add_patch(Rectangle( (x-size/2,y-size/2/6) ,  width=size , height=size/6 ,color=clr,fill=True, lw=0))
            self.ax.add_patch(Rectangle( (x-size/2/6,y-size/2) ,  width=size/6 , height=size ,color=clr,fill=True, lw=0))
        elif sign=="-":
            self.ax.add_patch(Rectangle( (x-size/2,y-size/2/6) ,  width=size , height=size/6 ,color=clr,fill=True, lw=0))

    def add_sign_annotations(self,add_sign_annotations=False,reference_group=None, method=None, stepsize=None):
        print(self.bbox)
        if not add_sign_annotations:
            return
        reference_data = np.max(reference_group.cell_data,axis=0)
        diffs = np.max(self.cell_data,axis=0) - reference_data
        # sign_col_center = self.cell_centers[0,-1] + self.cell_width #bump up once cell width to the right of the rightmost cell in the heatmap
        if method == "continuous":
            add_signs = lambda x,y,size: self._add_signs("+",x,y,abs(size),clr="black") if size>=0 else self._add_signs("-",x,y,abs(size),clr="black")

        elif method == "discrete":
            discrete_vals = np.arange(0,min(self.cell_width,self.cell_height),stepsize)
            threshold_size = lambda size: discrete_vals[int(np.sum(abs(size)>=discrete_vals)-1)]
            add_signs = lambda x,y,size: self._add_signs("+",x,y,threshold_size(size),clr="black") if size>=0 else self._add_signs("-",x,y,threshold_size(size),clr="black")
        elif method == "threshold":
            threshold = stepsize
            max_size = 0.6
            threshold_size = lambda size: min(self.cell_width,self.cell_height)*max_size if size>=threshold else 0
            add_signs = lambda x,y,size: self._add_signs("+",x,y,threshold_size(size),clr="black") if size>=0 else self._add_signs("-",x,y,threshold_size(size),clr="black")

        else:
            raise Exception("method should be chosen from 'continuous' or 'discrete'")

        for yi,y in enumerate(diffs):
            add_signs(self.cell_centers_x_arr[-1,yi]+self.cell_width,self.cell_centers_y_arr[-1,yi], diffs[yi])

        self.bbox[2]+=self.cell_width #increase 'width' in bounding box by cell_width


    def _add_signs(self,sign,x,y,size, clr="black"): #sign accepts "+","-"
        if sign=="+":
            self.ax.add_patch(Rectangle( (x-size/2,y-size/2/6) ,  width=size , height=size/6 ,color=clr,fill=True, lw=0))
            self.ax.add_patch(Rectangle( (x-size/2/6,y-size/2) ,  width=size/6 , height=size ,color=clr,fill=True, lw=0))
        elif sign=="-":
            self.ax.add_patch(Rectangle( (x-size/2,y-size/2/6) ,  width=size , height=size/6 ,color=clr,fill=True, lw=0))


    def add_colorbar_right(self, offset, width=None, height_scalefactor=1, boarder_width=None, boarder_clr="black", n_ticks=10, ticklabel_size=20,ticklabel_weight="bold",cbar_label=None, n_color_vals = 1000,span_height = False):
        hm_left, hm_bottom, hm_width, hm_height = self.heatmap_bbox

        width = width or self.cell_width #if width is None, set to self.cell_width
        height = hm_height*height_scalefactor
        boarder_width = boarder_width or self.bbwidth
        # boarder_clr = boarder_clr or "black"

        if self.vmin is None:
            self.vmin = np.min(self.cell_data)
        if self.vmax is None:
            self.vmax = np.max(self.cell_data)

        cb_left, cb_bottom, cb_width, cb_height = (hm_left+hm_width+offset,
                                                    hm_bottom+(hm_height/2)-height/2,
                                                    width,
                                                    height)  #left, bottom, width, height

        #add the boarder of the colorbar
        bd_left,bd_bottom,bd_width,bd_height = cb_left-boarder_width , cb_bottom-boarder_width , cb_width+boarder_width*2 , cb_height+boarder_width*2
        self.ax.add_patch(Rectangle( (bd_left, bd_bottom) ,  bd_width , bd_height ,color=boarder_clr,fill=True, lw=0))






        #add the colors to the colorbar (in n_color_vals discrete bands)
        colorband_height = cb_height/n_color_vals
        vmin_position = cb_bottom + colorband_height/2 #vmin
        vmax_position = cb_bottom+cb_height - colorband_height/2
        get_tick_position = lambda val: (val-self.vmin) * (vmax_position-vmin_position)/(self.vmax-self.vmin) + vmin_position



        color_positions = np.linspace(cb_bottom,cb_bottom+cb_height-colorband_height, n_color_vals) #y coordinates (in axis units) of the bottom of each color band
        color_vals = np.linspace(self.vmin,self.vmax, n_color_vals)
        for i in range(n_color_vals):
            clr = self.cmap(self.norm( color_vals[i] ))
            self.ax.add_patch(Rectangle( (cb_left, color_positions[i]) ,  cb_width , colorband_height ,color=clr,fill=True, lw=0))


        #y1 = mx1 + b
        #y2 = mx2 + b
        #b = y2-mx2  #vmax_position - (vmax_position-vmin_position)/(self.vmax-self.vmin) * self.vmax

        #add the ticks and tick labels - the following selects ticks with the fewest number of digits such that at least n_ticks ticks are added
        # n_ticks = 10
        tick_gap = 1
        while True:
            tick_min = np.ceil(self.vmin/tick_gap)*tick_gap
            increments = np.arange(tick_min,self.vmax, tick_gap)
            # print((len(increments)+1)*tick_gap + tick_min)
            # time.sleep(1)
            if (len(increments))*tick_gap+tick_min==self.vmax: #ad hoc method to make arange inclusive when it's possible to include both endpoints (np.arange(0.0,1.0,0.5) returns [0.0,0.5], not including the right endpoint even though it's possible with the given stepsize, so we append 0.1 manually )
                increments = np.concatenate([increments, np.array([self.vmax])])
                stride = (len(increments)-1)//(n_ticks-1)  #subtract 1 since we're adding the endpoint; the strides should be sized so that n_ticks-1 ticks are placed at the beginning of each stride, and the final right endpoint is placed to the right of the last strid (adding up to n_ticks ticks in total)
            else:
                stride = len(increments)//n_ticks
            n_strides = int(np.ceil(len(increments)/stride)) if stride>0 else -1
            if n_strides<n_ticks:
                tick_gap/=10
            else:
                ticks = [round(increments[i*stride],round(-np.log(tick_gap)/np.log(10))) for i in range(n_strides)]  #the rounding is to fix numerical errors in the np.arange result (the 1/tick_gap computes the number of decimal points possible with that gap; e.g. tick_gap=0.001, there are 3==-np.log(tick_gap)/np.log(10))
                break






        # tick_breadth = 0.05
        # tick_length = 0.1
        # tick_pad = 0.1
        # ticklabel_size=1
        # tick_positions = np.linspace(cb_bottom,cb_bottom+cb_height, len(ticks)) #y coordinates (in axis units) of the center of each tick
        # ticks_right=bd_left+bd_width+tick_length
        # print(ticks)
        # for tick_val in ticks:
        #     x = bd_left+bd_width
        #     y = get_tick_position(tick_val)
        #     self.ax.add_patch(Rectangle( (x, y-tick_breadth/2) ,  tick_length , tick_breadth ,color=boarder_clr,fill=True, lw=0))
        #
        #     tp,pt = self.get_text_patch(x+tick_length+tick_pad, y, str(tick_val), text_height=None, fontsize=ticklabel_size, weight=ticklabel_weight, alignment='r')
        #     self.ax.add_patch(pt)
        #     bb = tp.get_extents()
        #     print(bb)
        #     ticks_right = max(ticks_right,bb.xmax + tick_pad) #







        # tp,pt = self.get_text_patch(cb_bottom+cb_height/2, ticks_right, cbar_label, text_height=None, fontsize=ticklabel_size, weight=ticklabel_weight, alignment='r', rotation=90)
        # # tp = tp.transformed(Affine2D().rotate_deg(90))
        # self.ax.add_patch(pt)
        # bb = tp.get_extents()
        # print(bb)
        # # ticks_right = max(ticks_right,bb.xmax + tick_pad) #
        # cbar_label_right = bb.xmax



        #alternative ticks - added using typical mpl axis ticks/ticklabels
        axr = ax.twinx()
        for spine in axr.spines.values():
            spine.set_visible(False)
        # y1,y2 = ax.get_ylim()
        # axr.set_ylim(y1,y2)

        # axr.set_yticks([get_tick_position(tick) for tick in ticks])
        # axr.set_yticklabels([str(tick) for tick in ticks], size=ticklabel_size, weight=ticklabel_weight)

        # axr.set_ylim(y1,y2)
        ticks_right=bd_left+bd_width

        #alternative cbar label
        cbar_label_right=ticks_right
        # axr.tick_params(axis='both', pad=50)
        axr.set_ylabel(cbar_label, size=ticklabel_size, weight=ticklabel_weight, labelpad=10)












        #updating the self.bbox to include heatmap, title (if it was added), and colorbar
        old_right, old_top = self.bbox[0]+self.bbox[2] , self.bbox[1]+self.bbox[3]
        bd_right, bd_top  = bd_left+bd_width , bd_bottom+bd_height
        new_right = max(bd_right,ticks_right,cbar_label_right)
        bbox_left,bbox_bottom,bbox_right,bbox_top = min(self.bbox[0],bd_left), min(self.bbox[1],bd_bottom), max(old_right,new_right), max(old_top,bd_top)
        #new_bbox = (min(hm_left,bd_left), min(hm_bottom,bd_bottom), max(hm_width,bd_width), max(hm_height,bd_height)
        self.bbox = (bbox_left,bbox_bottom,bbox_right-bbox_left,bbox_top-bbox_bottom)
        self.colorbar_bbox = (cb_left, cb_bottom, cb_width, cb_height)





        return axr

#def group_width(self,group_len):
#    return (self.cell_width*group_len) + self.intragroup_spacing*(group_len-1)

#def group_left_extent(self,group_num):
#    return self.beginning_space + sum([self.group_width(group_len) for group_len in self.groups_of_cells[:group_num]]) + group_num*self.intergroup_spacing


#
# def get_cell_centers_old(group_data_grid, data, beginning_space = 0, cell_spacing = 1, intergroup_spacing = 0):
#
#     group_data_grid = group_data_grid.T[::-1]
#
#     cell_centers = {}
#     group_centers = {}
#     x_offset = beginning_space + cell_spacing/2
#     y_offset = beginning_space + cell_spacing/2
#     x_group_start = x_offset
#     y_group_start = y_offset
#
#     for x_group_idx in range(group_data_grid.shape[0]):
#         x_group_start = x_offset
#         for y_group_idx in range(group_data_grid.shape[1]):
#             y_group_start = y_offset
#             group = group_data_grid[x_group_idx,y_group_idx]
#             group_data = data[group][2]
#             cell_centers[group] = (np.arange(group_data.shape[0])*cell_spacing + y_offset, np.arange(group_data.shape[1])*cell_spacing + x_offset)
#             y_offset += intergroup_spacing + cell_spacing/2
#
#             group_centers[group] = ( (x_offset-x_group_start)/2, (y_offset-y_group_start)/2)
#         x_offset += intergroup_spacing + cell_spacing/2
#
#     return cell_centers, group_centers

def get_cell_centers(group_sizes, beginning_space = 0, cell_spacing = 1, intergroup_spacing = 0):

    cell_centers = {}
    group_centers = {}
    offset = beginning_space +cell_spacing/2
    for group,group_size in enumerate(group_sizes):
        group_start = offset
        cell_centers[group] = np.arange(group_size)*cell_spacing + offset
        offset += group_size*cell_spacing + intergroup_spacing

        group_centers[group] = (offset-group_start)/2

    return cell_centers, group_centers
    #   for cell_num in range(int(group_len)):
    #        cell_center = self.group_left_extent(group_num) + (self.cell_width/2) + cell_num*(self.intragroup_spacing + self.cell_width)
    #        cell_centers.append(cell_center)
    #return cell_centers






# class AnnotatedArray:
#     def __init__(self):
#         pass



#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/auto_subplots_adjust.html
import matplotlib.transforms as mtransforms

def fit_long_ylabels(fig,ax):
    def on_draw(event):
        bboxes = []
        for label in ax.get_yticklabels():
            # Bounding box in pixels
            bbox_px = label.get_window_extent()
            # Transform to relative figure coordinates. This is the inverse of
            # transFigure.
            bbox_fig = bbox_px.transformed(fig.transFigure.inverted())
            bboxes.append(bbox_fig)
        # the bbox that bounds all the bboxes, again in relative figure coords
        bbox = mtransforms.Bbox.union(bboxes)
        if fig.subplotpars.left < bbox.width:
            # Move the subplot left edge more to the right
            fig.subplots_adjust(left=1.1*bbox.width)  # pad a little
            fig.canvas.draw()


    fig.canvas.mpl_connect('draw_event', on_draw)
    # plt.show()

    # fig.draw()






##https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
from mpl_toolkits.axes_grid1 import Divider, Size
# def fix_axes_size_incm(fig,ax, axew, axeh):
#     axew = axew/2.54 #convert axew and axeh from cm to inches
#     axeh = axeh/2.54
#
#     #lets use the tight layout function to get a good padding size for our axes labels.
#     # fig = plt.gcf()
#     # ax = plt.gca()
#     fig.tight_layout()
#
#
#     #obtain the current ratio values for padding and fix size
#     oldw, oldh = fig.get_size_inches()
#     l = ax.figure.subplotpars.left
#     r = ax.figure.subplotpars.right
#     t = ax.figure.subplotpars.top
#     b = ax.figure.subplotpars.bottom
#
#     # gap=10
#     # print("[]]]]]]]]]]]]]]]]]]]]", l, r, t, b)
#     #work out what the new  ratio values for padding are, and the new fig size.
#     neww = axew+oldw*(1-r+l)
#     newh = axeh+oldh*(1-t+b)
#     newr = r*oldw/neww
#     newl = l*oldw/neww
#     newt = t*oldh/newh
#     newb = b*oldh/newh
#
#     #right(top) padding, fixed axes size, left(bottom) pading
#     hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Scaled(newl)]
#     vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]
#
#     divider = Divider(fig, (0.0, 0.0, 1., 1.), hori, vert, aspect=False)
#     # the width and height of the rectangle is ignored.
#
#     ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
#
#     #we need to resize the figure now, as we have may have made our axes bigger than in.
#     fig.set_size_inches(neww,newh)


def fix_axes_size_incm(fig, ax, axew, axeh, extra_left=0, extra_right= 0, extra_bottom = 0, extra_top = 0):
    """
    Adjust the size of the axes in cm and add extra space on the left for y-tick labels.

    Parameters:
    fig : Figure object
    ax : Axes object
    axew : float
        Width of the axes in cm
    axeh : float
        Height of the axes in cm
    extra_left : float, optional
        Extra space on the left in cm for y-tick labels (default is 1.0 cm)
    """
    if isinstance(ax, list):
        ax,twin_axes = ax[0],ax[1:]
    else:
        twin_axes = None

    axew = axew / 2.54  # convert axew and axeh from cm to inches
    axeh = axeh / 2.54
    extra_left = extra_left / 2.54  # convert extra_left from cm to inches

    fig.tight_layout()
    # fig.subplots_adjust(left=2)#0.9)

    oldw, oldh = fig.get_size_inches()
    #oldw+=extra_left
    l = ax.figure.subplotpars.left #+ extra_left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom



    # convert proportions from left side (basically cumulative proportions) to proportions for each individual section (left side, axis, right side)
    olpp = l #old left pad proportion
    orpp = 1-r #old right pad proportion
    oawp = 1 - (olpp+orpp) #old axis width proportion

    # convert proportions to sizes
    olps = oldw*olpp #old left pad size (inches)
    orps = oldw*orpp #old right pad size (inches)
    oaws = oldw*oawp # old axis width size (inches)

    # compute new size values ()
    naws = axew #new axis width size (inches)
    nlps = olps + extra_left #new left pad size (inches)
    nrps = orps + extra_right #new right pad size (inches)
    nfws = naws + nlps + nrps #new figure width size (inches)

    nlpp = nlps/nfws #new left pad proportion
    nrpp = nrps/nfws #new right pad proportion



    # convert proportions from bottom side (basically cumulative proportions) to proportions for each individual section (bottom pad, axis, top pad)
    obpp = b #old bottom pad proportion
    otpp = 1-t #old top pad proportion
    oahp = 1 - (obpp+otpp) #old axis height proportion

    # convert proportions to sizes
    obps = oldh*obpp #old bottom pad size (inches)
    otps = oldh*otpp #old top pad size (inches)
    oahs = oldh*oahp # old axis height size (inches)

    # compute new size values ()
    nahs = axeh #new axis height size (inches)
    nbps = obps + extra_bottom #new bottom pad size (inches)
    ntps = otps + extra_top #new top pad size (inches)
    nfhs = nahs + nbps + ntps #new figure height size (inches)

    nbpp = nbps/nfhs #new bottom pad proportion
    ntpp = ntps/nfhs #new top pad proportion




    #
    # neww = axew + oldw * (1 - r + l) + extra_left    #the fraction of oldw that is padding should stay fized as we adjust the axis size, so we add it to axew to get neww
    # newh = axeh + oldh * (1 - t + b)
    #
    # newr = r * oldw / neww #+ (extra_left / neww)   #r*oldw is the right side of ax in inches, divided by neww is the
    # newl = (l * oldw / neww) + (extra_left / neww)
    # newt = t * oldh / newh
    # newb = b * oldh / newh

    # hori = [Size.Scaled(nrpp), Size.Fixed(axew), Size.Scaled(nlpp)]
    # # hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Fixed(newl)]
    # vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]

    hori = [Size.Scaled(nlpp),Size.Fixed(axew),Size.Scaled(nrpp)]
    # hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Fixed(newl)]
    # vert = [Size.Scaled(newb),Size.Fixed(axeh),Size.Scaled(newt)]
    vert = [Size.Scaled(nbpp),Size.Fixed(axeh),Size.Scaled(ntpp)]


    # fig.set_size_inches(neww, newh)
    divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), hori, vert, aspect=False)

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
    if not twin_axes is None:
        for t_ax in twin_axes:
            t_ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    fig.set_size_inches(nfws, nfhs)
#
# def fix_axes_size_incm_twinaxes(fig,axes, axew, axeh):
#     axew = axew/2.54
#     axeh = axeh/2.54
#
#     #lets use the tight layout function to get a good padding size for our axes labels.
#     # fig = plt.gcf()
#     # ax = plt.gca()
#     fig.tight_layout()
#     #obtain the current ratio values for padding and fix size
#     oldw, oldh = fig.get_size_inches()
#     l = ax.figure.subplotpars.left
#     r = ax.figure.subplotpars.right
#     t = ax.figure.subplotpars.top
#     b = ax.figure.subplotpars.bottom
#
#     #work out what the new  ratio values for padding are, and the new fig size.
#     neww = axew+oldw*(1-r+l)
#     newh = axeh+oldh*(1-t+b)
#     newr = r*oldw/neww
#     newl = l*oldw/neww
#     newt = t*oldh/newh
#     newb = b*oldh/newh
#
#     #right(top) padding, fixed axes size, left(bottom) pading
#     hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Scaled(newl)]
#     vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]
#
#     divider = Divider(fig, (0.0, 0.0, 1., 1.), hori, vert, aspect=False)
#     # the width and height of the rectangle is ignored.
#
#     ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
#
#     #we need to resize the figure now, as we have may have made our axes bigger than in.
#     fig.set_size_inches(neww,newh)
#








def add_colorbar_right(heatmap_bbox, fig_bbox, offset, vmin, vmax, cell_width, bbwidth, ax, cmap, norm, width=None, height_scalefactor=1, boarder_width=None, boarder_clr="black", n_ticks=10, ticklabel_size=20,ticklabel_weight="bold",cbar_label=None, n_color_vals = 1000,span_height = False):
    hm_left, hm_bottom, hm_width, hm_height = heatmap_bbox

    # vmin, vmax, cell_width, bbwidth, ax, cmap, norm

    width = width or cell_width #if width is None, set to self.cell_width
    height = hm_height*height_scalefactor
    boarder_width = boarder_width or bbwidth
    # boarder_clr = boarder_clr or "black"


    cb_left, cb_bottom, cb_width, cb_height = (hm_left+hm_width+offset,
                                                hm_bottom+(hm_height/2)-height/2,
                                                width,
                                                height)  #left, bottom, width, height
    print(cb_left, cb_bottom, cb_width, cb_height)
    print(heatmap_bbox)
    print(fig_bbox)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    #add the boarder of the colorbar
    bd_left,bd_bottom,bd_width,bd_height = cb_left-boarder_width , cb_bottom-boarder_width , cb_width+boarder_width*2 , cb_height+boarder_width*2
    ax.add_patch(Rectangle( (bd_left, bd_bottom) ,  bd_width , bd_height ,color=boarder_clr,fill=True, lw=0))






    #add the colors to the colorbar (in n_color_vals discrete bands)
    colorband_height = cb_height/n_color_vals
    vmin_position = cb_bottom + colorband_height/2 #vmin
    vmax_position = cb_bottom+cb_height - colorband_height/2
    get_tick_position = lambda val: (val-vmin) * (vmax_position-vmin_position)/(vmax-vmin) + vmin_position



    color_positions = np.linspace(cb_bottom,cb_bottom+cb_height-colorband_height, n_color_vals) #y coordinates (in axis units) of the bottom of each color band
    color_vals = np.linspace(vmin,vmax, n_color_vals)
    # color_positions = np.array([get_tick_position(v) for v in color_vals])
    for i in range(n_color_vals):
        clr = cmap(norm( color_vals[i] ))
        ax.add_patch(Rectangle( (cb_left, color_positions[i]) ,  cb_width , colorband_height ,color=clr,fill=True, lw=0))
        # ax.scatter(cb_left, color_positions[i])
        # print(cb_left, color_positions[i])
        # print(ax.get_ylim())

    #y1 = mx1 + b
    #y2 = mx2 + b
    #b = y2-mx2  #vmax_position - (vmax_position-vmin_position)/(self.vmax-self.vmin) * self.vmax

    #add the ticks and tick labels - the following selects ticks with the fewest number of digits such that at least n_ticks ticks are added
    # n_ticks = 10
    tick_gap = 1
    while True:
        tick_min = np.ceil(vmin/tick_gap)*tick_gap
        increments = np.arange(tick_min,vmax, tick_gap)
        # print((len(increments)+1)*tick_gap + tick_min)
        # time.sleep(1)
        if (len(increments))*tick_gap+tick_min==vmax: #ad hoc method to make arange inclusive when it's possible to include both endpoints (np.arange(0.0,1.0,0.5) returns [0.0,0.5], not including the right endpoint even though it's possible with the given stepsize, so we append 0.1 manually )
            increments = np.concatenate([increments, np.array([vmax])])
            stride = (len(increments)-1)//(n_ticks-1)  #subtract 1 since we're adding the endpoint; the strides should be sized so that n_ticks-1 ticks are placed at the beginning of each stride, and the final right endpoint is placed to the right of the last strid (adding up to n_ticks ticks in total)
        else:
            stride = len(increments)//n_ticks
        n_strides = int(np.ceil(len(increments)/stride)) if stride>0 else -1
        if n_strides<n_ticks:
            tick_gap/=10
        else:
            round_decimal = round(float(-np.log(tick_gap)/np.log(10)))
            if round_decimal==0: #annoyingly, it seems that python's "round" function keeps the input as a float even if rounding to 0 decimals (round(x,0)) when the input is a float; if you just do round(x) it converts it to an int, but not when you do round(x,0)
                ticks = [round(float(increments[i*stride])) for i in range(n_strides)]  #the rounding is to fix numerical errors in the np.arange result (the 1/tick_gap computes the number of decimal points possible with that gap; e.g. tick_gap=0.001, there are 3==-np.log(tick_gap)/np.log(10))
            else:
                ticks = [round(float(increments[i*stride]), round_decimal) for i in range(n_strides)]  #the rounding is to fix numerical errors in the np.arange result (the 1/tick_gap computes the number of decimal points possible with that gap; e.g. tick_gap=0.001, there are 3==-np.log(tick_gap)/np.log(10))


            # print(ticks)
            # print(round(-np.log(tick_gap)/np.log(10)))
            # print([round(t,0) for t in ticks])
            break



    # print(ticks)



    # tick_breadth = 0.05
    # tick_length = 0.1
    # tick_pad = 0.1
    # ticklabel_size=1
    # tick_positions = np.linspace(cb_bottom,cb_bottom+cb_height, len(ticks)) #y coordinates (in axis units) of the center of each tick
    # ticks_right=bd_left+bd_width+tick_length
    # print(ticks)
    # for tick_val in ticks:
    #     x = bd_left+bd_width
    #     y = get_tick_position(tick_val)
    #     self.ax.add_patch(Rectangle( (x, y-tick_breadth/2) ,  tick_length , tick_breadth ,color=boarder_clr,fill=True, lw=0))
    #
    #     tp,pt = self.get_text_patch(x+tick_length+tick_pad, y, str(tick_val), text_height=None, fontsize=ticklabel_size, weight=ticklabel_weight, alignment='r')
    #     self.ax.add_patch(pt)
    #     bb = tp.get_extents()
    #     print(bb)
    #     ticks_right = max(ticks_right,bb.xmax + tick_pad) #







    # tp,pt = self.get_text_patch(cb_bottom+cb_height/2, ticks_right, cbar_label, text_height=None, fontsize=ticklabel_size, weight=ticklabel_weight, alignment='r', rotation=90)
    # # tp = tp.transformed(Affine2D().rotate_deg(90))
    # self.ax.add_patch(pt)
    # bb = tp.get_extents()
    # print(bb)
    # # ticks_right = max(ticks_right,bb.xmax + tick_pad) #
    # cbar_label_right = bb.xmax



    #alternative ticks - added using typical mpl axis ticks/ticklabels
    # print(ax.get_ylim())

    axr = ax.twinx()
    # axr=ax

    for spine in axr.spines.values():
        spine.set_visible(False)
    # print(axr.get_ylim(), ax.get_ylim())
    # ax.set_ylim([0,5])
    # axr.scatter(1,1,c='r')
    # y1,y2 = ax.get_ylim()
    # axr.set_ylim(y1,y2)
    # color_positions = np.array([get_tick_position(v) for v in color_vals])
    # ax.scatter(np.ones(color_positions.shape), color_positions)
    # ax.scatter(1,get_tick_position(max(color_vals)))
    # ax.scatter(2,get_tick_position(max(ticks)))

    # axr.set_yticks([.2,.3,.4])
    # axr.set_yticklabels([.2,.3,.4])
    axr.set_yticks([get_tick_position(tick) for tick in ticks])
    # axr.set_yticks([1 for tick in ticks])
    axr.set_yticklabels([str(tick) for tick in ticks], size=ticklabel_size, weight=ticklabel_weight)

    # axr.set_ylim(y1,y2)
    ticks_right=bd_left+bd_width

    #alternative cbar label
    cbar_label_right=ticks_right
    # axr.tick_params(axis='both', pad=50)
    axr.set_ylabel(cbar_label, size=ticklabel_size, weight=ticklabel_weight, labelpad=10)












    # updating the self.bbox to include heatmap, title (if it was added), and colorbar
    old_right, old_top = fig_bbox[0]+fig_bbox[2] , fig_bbox[1]+fig_bbox[3]
    bd_right, bd_top  = bd_left+bd_width , bd_bottom+bd_height
    new_right = max(bd_right,ticks_right,cbar_label_right)
    bbox_left,bbox_bottom,bbox_right,bbox_top = min(fig_bbox[0],bd_left), min(fig_bbox[1],bd_bottom), max(old_right,new_right), max(old_top,bd_top)
    #new_bbox = (min(hm_left,bd_left), min(hm_bottom,bd_bottom), max(hm_width,bd_width), max(hm_height,bd_height)
    fig_bbox = (bbox_left,bbox_bottom,bbox_right-bbox_left,bbox_top-bbox_bottom)
    colorbar_bbox = (cb_left, cb_bottom, cb_width, cb_height)
    # colorbar_bbox=None




    return axr, fig_bbox, colorbar_bbox














def collect_ticks(data_group_grid, all_cell_centers):
    all_x_cell_centers={g:cc[0] for g,cc in all_cell_centers.items()}
    all_y_cell_centers={g:cc[1] for g,cc in all_cell_centers.items()}
    # all_x_cell_centers,all_y_cell_centers = {g:zip(*[all_cell_centers[g] for g in all_cell_centers.keys()])
    xticks = np.concatenate([all_x_cell_centers[g] for g in data_group_grid[0,:]],axis=0)
    yticks = np.concatenate([all_y_cell_centers[g] for g in data_group_grid[:,0]],axis=0)
    return xticks,yticks


def collect_labels(data_group_grid, all_x_labels, all_y_labels):
    xlabels = np.concatenate([all_x_labels[g] for g in data_group_grid[0,:]],axis=0)
    ylabels = np.concatenate([all_y_labels[g] for g in data_group_grid[:,0]],axis=0)
    return xlabels,ylabels



# def custom_imshow(fig,ax,data, data_group_grid, bar_spans_data,xlabels,ylabels,group_labels,pad_val,cmap,ylabel_offset=None,xlabel_offset=None,colorbar_offset=None,title_offset=None,fig_scale=None,vmin=None,vmax=None, origin="upper",add_highlights=True, group_sizes = None):
def custom_imshow(fig,ax,data, data_group_grid,cmap,cbar_label=None,vmin=None,vmax=None,add_highlights=True, group_sizes = None, add_text=False, add_group_titles = True, group_title_fontsize=0.8, axislabel_size=15,cell_size=0.7, tcolor='white', add_sign_annotations=False, reference_group=None, sign_annotation_method="continuous", signs_annotation_discrete_stepsize=0.1):

    for spine in ax.spines.values():
        spine.set_visible(False)


    data_group_grid = data_group_grid[::-1, :]
    if add_highlights: #reverse y_labels and rows of data (then transpose data so that y~row, x~col to make indexing consistant )
        data = {group_name: (x_labels, y_labels[::-1], data_arr_2D[::-1].T, highlights_mask[::-1].T) for group_name,(x_labels, y_labels, data_arr_2D, highlights_mask) in data.items()}
    else:
        data = {group_name: (x_labels, y_labels[::-1], data_arr_2D[::-1].T, np.zeros(data_arr_2D[::-1].T.shape)) for group_name, (x_labels, y_labels, data_arr_2D, _) in data.items()} #make the mask all zeros with the same shape as the cell data

    cell_width=cell_size
    cell_height=cell_size#cell_width
    bbwidth=0.1
    wbwidth=0.1


    # vmin=0
    # vmax=1
    intergroup_spacing =.3

    data_group_grid = data_group_grid.T#[::-1]



    if isinstance(group_sizes,type(None)):
        group_sizes_x = [data[group][2].shape[0] for group in data_group_grid[:,0]]
        group_sizes_y = [data[group][2].shape[1] for group in data_group_grid[0,:]]


    cell_centers_x,group_centers_x = get_cell_centers(group_sizes_x, beginning_space = 0, cell_spacing = cell_width, intergroup_spacing = intergroup_spacing)
    cell_centers_y,group_centers_y = get_cell_centers(group_sizes_y, beginning_space = 0, cell_spacing = cell_height, intergroup_spacing = intergroup_spacing)



    cell_centers = {data_group_grid[i,j]:(cell_centers_x[i],cell_centers_y[j]) for i in range(data_group_grid.shape[0]) for j in range(data_group_grid.shape[1])}
    group_centers = {data_group_grid[i,j]:(group_centers_x[i],group_centers_y[j]) for i in range(data_group_grid.shape[0]) for j in range(data_group_grid.shape[1])}


    xticks,yticks = collect_ticks(data_group_grid.T, cell_centers)
    all_x_labels,all_y_labels = {g:data[g][0] for g in data.keys()}, {g:data[g][1] for g in data.keys()}
    xlabels,ylabels = collect_labels(data_group_grid.T, all_x_labels, all_y_labels)

    groupxticks = np.array([group_centers[g][0] for g in data_group_grid.T[0,:]])
    groupxlabels = data_group_grid.T[0,:]



    fig_left,fig_bottom,fig_right,fig_top = (0,0,0,0)
    hm_left,hm_bottom,hm_right,hm_top = (0,0,0,0)
    heatmap_objs = {}
    for gi,group in enumerate(data.keys()):

        group_cell_centers = cell_centers[group]
        group_cell_data = data[group][2]
        group_mask = data[group][3]

        # print("=================")
        highlight_builder = MergedHighlightsImshow(fig, ax, group_cell_centers, group_cell_data, cell_width, cell_height, bbwidth, wbwidth, cmap, vmin, vmax)
        # print(highlight_builder.bbox)
        heatmap_objs[group]=highlight_builder
        highlight_builder.add_all_highlights(group_mask, bbwidth, wbwidth, merge_along_x_axis=True, merge_along_y_axis=False, add_text=add_text, tcolor=tcolor)
        # print(highlight_builder.bbox)

        if add_group_titles:
            highlight_builder.add_title(group, horizontalalignment='center', verticalalignment='bottom', fontsize=group_title_fontsize,weight="bold") #should add text above the heatmap, and then adjust the bbox of the highlight_builder

        # print(highlight_builder.bbox)

        reference = heatmap_objs[reference_group] if add_sign_annotations else None
        highlight_builder.add_sign_annotations(add_sign_annotations=add_sign_annotations,reference_group=reference, method=sign_annotation_method, stepsize=signs_annotation_discrete_stepsize)
        # print(highlight_builder.bbox)

        # ax.add_patch(Rectangle( (highlight_builder.bbox[0],highlight_builder.bbox[1]) ,  highlight_builder.bbox[2] , highlight_builder.bbox[3] ,color="green",fill=False, lw=1))

        #now add the colorbar to the right for the last heatmap
        # if group in data_group_grid[-1,:]: #if the current group is in the rightmost column
        # cbar_offset = intergroup_spacing
        # cbar_height_scalefactor = 1#0.8
        # cbar_label="#trials"
        # axr = highlight_builder.add_colorbar_right(cbar_offset, width=None, height_scalefactor=cbar_height_scalefactor, boarder_width=None, boarder_clr="black", n_ticks=10, ticklabel_size=axislabel_size,ticklabel_weight="bold",cbar_label=cbar_label, n_color_vals = 1000, span_height = True)

        bbox = highlight_builder.bbox
        hmbbox = highlight_builder.heatmap_bbox
        fig_left,fig_bottom,fig_right,fig_top = min(fig_left,bbox[0]), min(fig_bottom,bbox[1]), max(fig_right,bbox[0]+bbox[2]), max(fig_top,bbox[1]+bbox[3])
        hm_left,hm_bottom,hm_right,hm_top = min(hm_left,hmbbox[0]), min(hm_bottom,hmbbox[1]), max(hm_right,hmbbox[0]+hmbbox[2]), max(hm_top,hmbbox[1]+hmbbox[3])

    cbar_offset = intergroup_spacing
    cbar_height_scalefactor = 0.8
    cbar_label="#trials"
    heatmap_bbox = (hm_left,hm_bottom,hm_right-hm_left,hm_top-hm_bottom)
    fig_bbox = (fig_left,fig_bottom,fig_right,fig_top)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    axr, fig_bbox, cbar_bbox = add_colorbar_right(heatmap_bbox, fig_bbox, cbar_offset, vmin, vmax, cell_width, bbwidth, ax, cmap, norm, width=None, height_scalefactor=cbar_height_scalefactor, boarder_width=None, boarder_clr="black", n_ticks=3, ticklabel_size=axislabel_size,ticklabel_weight="bold",cbar_label=cbar_label, n_color_vals = 1000, span_height = True)
    bbox = highlight_builder.bbox
    fig_left,fig_bottom,fig_right,fig_top = fig_bbox #min(fig_left,bbox[0]), min(fig_bottom,bbox[1]), max(fig_right,bbox[0]+bbox[2]), max(fig_top,bbox[1]+bbox[3])

    # ax.scatter(fig_left, fig_bottom)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xlabels, fontsize=axislabel_size, weight="bold", rotation=25)
    ax.set_yticklabels(ylabels, fontsize=axislabel_size, weight="bold", rotation=0)

    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # height=fig_top-fig_bottom
    # fig_left,fig_right,fig_top,fig_bottom = add_colorbar_right(ax,fig_left,fig_right,fig_bottom,fig_top, width=1, height=height, cmap=cmap, norm=norm) #updates fig_right
    # ax.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))

    ax.set_xlim(fig_left,fig_right)
    ax.set_ylim(fig_bottom,fig_top)


    axr.set_xlim(fig_left,fig_right)
    axr.set_ylim(fig_bottom,fig_top)


    fix_axes_size_incm(fig,[ax,axr], fig_right-fig_left, fig_top-fig_bottom, extra_left=0)

















# def custom_imshow(fig,ax,data, data_group_grid, bar_spans_data,xlabels,ylabels,group_labels,pad_val,cmap,ylabel_offset=None,xlabel_offset=None,colorbar_offset=None,title_offset=None,fig_scale=None,vmin=None,vmax=None, origin="upper",add_highlights=True, group_sizes = None):
def custom_imshow_lower_triangle_heatmap(fig,ax,data, data_group_grid,cmap,cbar_label=None,vmin=None,vmax=None,add_highlights=True, group_sizes = None, add_text=False, add_group_titles = True, group_title_fontsize=0.8, axislabel_size=15,cell_size=0.7, tcolor='white', add_sign_annotations=False, reference_group=None, sign_annotation_method="continuous", signs_annotation_discrete_stepsize=0.1):

    for spine in ax.spines.values():
        spine.set_visible(False)


    if True: #reverse y_labels and rows of data (then transpose data so that y~row, x~col to make indexing consistant )
        data = {group_name: (x_labels, y_labels[::-1], data_arr_2D[::-1].T, highlights_mask[::-1].T) for group_name,(x_labels, y_labels, data_arr_2D, highlights_mask) in data.items()}
    else:
        data = {group_name: (x_labels, y_labels[::-1], data_arr_2D[::-1].T, np.zeros(data_arr_2D[::-1].T.shape)) for group_name, (x_labels, y_labels, data_arr_2D, _) in data.items()} #make the mask all zeros with the same shape as the cell data

    cell_width=cell_size
    cell_height=cell_size#cell_width
    bbwidth=0.1
    wbwidth=0.1


    # vmin=0
    # vmax=1
    intergroup_spacing =.1

    data_group_grid = data_group_grid.T#[::-1]



    if isinstance(group_sizes,type(None)):
        group_sizes_x = [data[group][2].shape[0] for group in data_group_grid[:,0]]
        group_sizes_y = [data[group][2].shape[1] for group in data_group_grid[0,:]]


    cell_centers_x,group_centers_x = get_cell_centers(group_sizes_x, beginning_space = 0, cell_spacing = cell_width, intergroup_spacing = intergroup_spacing)
    cell_centers_y,group_centers_y = get_cell_centers(group_sizes_y, beginning_space = 0, cell_spacing = cell_height, intergroup_spacing = intergroup_spacing)



    cell_centers = {data_group_grid[i,j]:(cell_centers_x[i],cell_centers_y[j]) for i in range(data_group_grid.shape[0]) for j in range(data_group_grid.shape[1])}
    group_centers = {data_group_grid[i,j]:(group_centers_x[i],group_centers_y[j]) for i in range(data_group_grid.shape[0]) for j in range(data_group_grid.shape[1])}


    xticks,yticks = collect_ticks(data_group_grid.T, cell_centers)
    all_x_labels,all_y_labels = {g:data[g][0] for g in data.keys()}, {g:data[g][1] for g in data.keys()}
    xlabels,ylabels = collect_labels(data_group_grid.T, all_x_labels, all_y_labels)

    groupxticks = np.array([group_centers[g][0] for g in data_group_grid.T[0,:]])
    groupxlabels = data_group_grid.T[0,:]



    fig_left,fig_bottom,fig_right,fig_top = (0,0,0,0)
    heatmap_objs = {}
    for gi,group in enumerate(data.keys()):

        group_cell_centers = cell_centers[group]
        group_cell_data = data[group][2]
        group_mask = data[group][3]


        highlight_builder = MergedHighlightsImshow(fig, ax, group_cell_centers, group_cell_data, cell_width, cell_height, bbwidth, wbwidth, cmap, vmin, vmax)
        heatmap_objs[group]=highlight_builder

        boarder_mask=group_mask
        group_mask=np.zeros(group_mask.shape)
        highlight_builder.make_lower_triangle_heatmap(group_mask, boarder_mask, bbwidth, wbwidth, merge_along_x_axis=False, merge_along_y_axis=False, add_text=add_text, tcolor=tcolor)
        if add_group_titles:
            highlight_builder.add_title(group, horizontalalignment='center', verticalalignment='bottom', fontsize=group_title_fontsize,weight="bold") #should add text above the heatmap, and then adjust the bbox of the highlight_builder

        reference = heatmap_objs[reference_group] if add_sign_annotations else None
        highlight_builder.add_sign_annotations(add_sign_annotations=add_sign_annotations,reference_group=reference, method=sign_annotation_method, stepsize=signs_annotation_discrete_stepsize)

        #now add the colorbar to the right for the last heatmap
        if group in data_group_grid[-1,:]: #if the current group is in the rightmost column
            cbar_offset = intergroup_spacing
            cbar_height_scalefactor = 1#0.8
            cbar_label="Spearman Correlation"
            axr = highlight_builder.add_colorbar_right(cbar_offset, width=None, height_scalefactor=cbar_height_scalefactor, boarder_width=None, boarder_clr="black", n_ticks=3, ticklabel_size=axislabel_size,ticklabel_weight="bold",cbar_label=cbar_label, n_color_vals = 1000)
        bbox = highlight_builder.bbox
        fig_left,fig_bottom,fig_right,fig_top = min(fig_left,bbox[0]), min(fig_bottom,bbox[1]), max(fig_right,bbox[0]+bbox[2]), max(fig_top,bbox[1]+bbox[3])


    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xlabels, fontsize=axislabel_size, weight="bold", rotation=90)
    ax.set_yticklabels(ylabels, fontsize=axislabel_size, weight="bold", rotation=0)

    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # height=fig_top-fig_bottom
    # fig_left,fig_right,fig_top,fig_bottom = add_colorbar_right(ax,fig_left,fig_right,fig_bottom,fig_top, width=1, height=height, cmap=cmap, norm=norm) #updates fig_right
    # ax.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))

    ax.set_xlim(fig_left,fig_right)
    ax.set_ylim(fig_bottom,fig_top)


    axr.set_xlim(fig_left,fig_right)
    axr.set_ylim(fig_bottom,fig_top)


    fix_axes_size_incm(fig,[ax,axr], fig_right-fig_left, fig_top-fig_bottom, extra_left=1)





#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^





