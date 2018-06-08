procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
      z : Integer;
   end record;

   function F(x : Integer) return Integer;

   p : Point := (2, 3, 12);
begin
   p.y := F(p.x);
end Ex1;
