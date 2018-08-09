procedure Test is
   type Point is record
      x : Integer;
      y : Integer;
   end record;

   type Rectangle is record
      center : Point;
      width : Integer;
      height : Integer;
   end record;

   p : access Integer;
   r : Rectangle;
begin
   p := r.center.x'Access;
   if p.all = 2 then
      r.width := p.all + 2;
   end if;
end Ex1;
