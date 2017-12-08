procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
   end record;
   type Rectangle is record
      top_left: access Point;
      bottom_right: access Point;
   end record;
   r : Rectangle;
   x : Integer;
begin
   x := r.top_left.x;
end Ex1;
