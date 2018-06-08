procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
      z : Integer;
   end record;

   function F(x : Integer) return Integer is
   begin
      return x + 1;
   end F;

   p : Point := (2, 3, 12);
begin
   p.x := F(2);
end Ex1;
