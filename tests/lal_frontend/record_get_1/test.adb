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
   z : Boolean := False;
begin
   if (r.top_left.x = 2 and
       r.top_left.y = 3 and
       r.bottom_right.x = 12 and
       r.bottom_right.y = 8) then
      z := True;
   end if;
end Ex1;
