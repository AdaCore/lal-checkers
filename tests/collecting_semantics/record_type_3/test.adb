procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
   end record;
   type Rectangle is record
      top_left: Point;
      bottom_right: Point;
   end record;
   r : Rectangle;
   i : Integer := 12;
begin
   r.top_left.x := i;
   r.top_left.y := i - 2;
   r.bottom_right.x := i + 5;
   r.bottom_right.y := i + 10;
end Ex1;
