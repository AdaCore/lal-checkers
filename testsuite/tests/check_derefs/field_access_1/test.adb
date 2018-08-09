procedure Test is
   type Point is record
      x : Integer;
      y : Integer;
   end record;
   type Point_Access is access Point;
   type Rectangle is record
      top_left: Point_Access;
      bottom_right: Point_Access;
   end record;
   r : Rectangle;
   p : Point := (1, 2);
   x : Integer;
   b : Boolean;
begin
   if b then
      r.top_left := p'Access;
   end if;

   x := r.top_left.x;
end Ex1;
